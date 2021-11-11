# Deep Multi-Agent Racing
Tensorised verison of the OpenAI gym car racing environment
(image rendering is not yet supported for more than one environment at once)
![plot](media/env.png )
## Setting up the Environment: 
1. get anaconda (e.g. https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/)
2. activate conda in shell
3. run (from base directory of repo) ```bash setup_conda_env.sh```
4. Install

    ```pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html```
5. Get the rl library, in same terminal run:

    ```cd .. && git clone git@github.com:leggedrobotics/rsl_rl.git```

    ```cd rsl-rl && pip install -e . ```

6. Ready to go! Try:

    ```python scripts/playManual.py```  


## Viewer Controls

* V     : Toggle rendering
* Q     : Quit
* RT    : Switch between environments
* WASD  : Move camera
* ,.    : Zoom camera