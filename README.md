# Email Message Processing and Analysis

Research code for processing and analysing email and newsgroup messages.

Install dependencies via:

    pip3 install -r requirements.txt
    
The `run.sh` script from the main `src` can be used to start any of the tools
and services with the correct `PYTHONPATH`.

A web UI for data exploration can be found in `src/explorer/explorer.py`.
Before starting it, copy the main config file `src/explorer/conf/settings.py` to
`src/explorer/conf/local_settings.py` and adjust the config values (e.g. set the
correct model paths etc.)
