import launch
if not launch.is_installed("matplotlib"):
    launch.run_pip("install matplotlib", "requirements for depthmap script")