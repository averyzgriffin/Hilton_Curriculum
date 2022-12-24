import os
from os.path import exists
import torch

from torch.optim.lr_scheduler import LambdaLR
# import GPUtil
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from Chapter1_Transformers.models import make_gpt
from data_loader import create_dataloaders, load_tokenizers, load_vocab
from model_arch import make_decoder_model
from training_setup import LabelSmoothing, rate,run_epoch, Batch,\
    DummyOptimizer, DummyScheduler, SimpleLossCompute, TrainState,\
    BatchDecoder


def train_worker(
    gpu,
    ngpus_per_node,
    # vocab_src,
    vocab_tgt,
    # spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512  # TODO parameter
    # model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    # model = make_decoder_model(len(vocab_tgt), N=6, d_model=d_model)
    model = make_gpt(len(vocab_tgt), N=6, d_model=d_model, h=8)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        # vocab_src,
        vocab_tgt,
        # spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            # (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            (BatchDecoder(b, pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        # GPUtil.showUtilization()

        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        model.eval()
        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        sloss = run_epoch(
            # (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            (BatchDecoder(b, pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )

        print(sloss)
        torch.cuda.empty_cache()

    # if is_main_process:  TODO add back in later
    #     file_path = "%sfinal.pt" % config["file_prefix"]
    #     torch.save(module.state_dict(), file_path)


# def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
def train_distributed_model(vocab_tgt, spacy_en, config):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        # args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
        args=(ngpus, vocab_tgt, spacy_en, config, True),
    )


# def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
def train_model(vocab_tgt, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            # vocab_src, vocab_tgt, spacy_de, spacy_en, config
            vocab_tgt, spacy_en, config
        )
    else:
        train_worker(
            # 0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
            0, 1, vocab_tgt, spacy_en, config, False
        )


def load_trained_model(spacy_en, vocab_tgt):
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "GPTAve_model_",
    }
    model_path = "GPTAve_model_final.pt"
    if not exists(model_path):
        # train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
        train_model(vocab_tgt, spacy_en, config)

    # model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    # model = make_decoder_model(len(vocab_tgt), N=6)
    model = make_gpt(len(vocab_tgt), N=6, d_model=512, h=8)
    model.load_state_dict(torch.load("GPTAve_model_final.pt"))  # TODO Saving
    return model


if __name__ == "__main__":
    spacy_de, spacy_en = load_tokenizers()
    # vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    vocab_tgt = load_vocab(spacy_en)

    load_trained_model(spacy_en, vocab_tgt)













