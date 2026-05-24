def pytest_addoption(parser):
    group = parser.getgroup("manual")
    group.addoption("--run-manual", action="store_true", help="run manually gated GPU/model smoke tests")
    group.addoption("--model-path", action="store", default=None, help="DiT checkpoint path for manual model smoke tests")
    group.addoption("--model-version", action="store", default="klein-base-9b", help="FLUX.2 model version for manual smoke")
    group.addoption("--blocks-to-swap", action="store", type=int, default=4, help="block-swap count for manual smoke")
    group.addoption("--compile-blocks", action="store_true", help="torch.compile transformer blocks in manual smoke")
