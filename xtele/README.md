# When first use:
## 1. Install pip-pkg, config file, cmd-tool;

    git clone http://10.0.3.101/embodied-ai/sysEAI/xtele.git
    cd xtele/xtele/scripts
    bash install_all.sh

## 2. Modify the config file.
Open the config file by cmd-lines

    xhumanoid-xtele --openconfig

A Tienkung config is default installed.
Replace the serial port by local serial-port name.

    ls /dev/serial/by-id/*

You can also install other config template.
e.g. noitom config:

    cd xtele/xtele/scripts
    bash install_config.sh noitom


# Calibrate the tele-operation table.
Using cali-tool and follow the terminal text prompt.

    xhumanoid-xtele --mode cali

# Get current state

    xhumanoid-xtele --mode getstates

# Check the system status and find the problem

    xhumanoid-xtele --mode systemtest

# Check the sign of the producted tele-arm. (Currently only supports Tienkung2)

    xhumanoid-xtele --mode checksign
