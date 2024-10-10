# ML Tools for Precision Physics in HEP

Welcome to the experimental section of the Physics at Colliders 2024 PhD Course (Milano-Bicocca). 

- Indico agenda: https://indico.cern.ch/event/1466371/
- Repository: https://github.com/valsdav/PhDCourse_MLForPrecisionPhysics_2024
- Dataset and code:  https://cernbox.cern.ch/files/spaces/eos/user/d/dvalsecc/EFT_PhD_Course
- Dataset WW VBS features [plots](https://dvalsecc.web.cern.ch/dvalsecc/EFT_PhD_Course_2024/plots/WW_sample_plots/vbs_sel_looser/)
   
## Table of content

- Dataset preparation
    - features scaling and normalization
    - data manipulation and formatting

- Transformers
    - Intro and architecture
    - Full particles regression with transformers
      - Best losses for full particle regression
      - Constrained optimization with MDMM

- Normalizing Flows:
   - Intro and architecture
   - Example: Conditional probability for event boost
   - Application: Generative Transformers for neutrinos generation

## Setup
### Setup at CERN
```bash
# Open a connection to lxplus-gpu with a port-forwarding on 8888 to visualize jupyter notebook
ssh -L 8888:localhost:8888 lxplus-gpu.cern.ch
# optionally move to eos to have more disk space
# cd /eos/user/your/name
mkdir PhDCourse2024
cd PhDCourse2024

# Let's use tmux to keep the session open, note down your lxplus-gpu hostname
tmux new -t course

# Start the apptainer shalle
apptainer shell -B ${XDG_RUNTIME_DIR} \
          --nv -B /afs -B /cvmfs/cms.cern.ch \
          -B /eos/user/d/dvalsecc/PhDCourse_MLColliderPhysics2024 \
          --bind /tmp  -B /eos/user/your/your-user \
          --bind /etc/sysconfig/ngbauth-submit  \
          --env KRB5CCNAME=${XDG_RUNTIME_DIR}/krb5cc \
          /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.11-cuda

# Now from inside the singularity we create a virtual env to install some additional packages
python -m venv myenv --system-site-packages

# Activate the environment TO BE DONE ALL THE TIME
source myenv/bin/activate
# install packages (to doonly once)
python -m pip install -r requirements.txt

# Make the virtualenv visible to jupyter lab
python -m ipykernel install --user --name=myenv

# Now we can start the jupyter notebook, 
jupyter lab
```

#### Setup outside CERN
We don't need special software apart from torch (with CUDA support possibly). 

You can use docker or apptainer to have a basic python environment and them install the required packages on top.

```bash

docker run --gpus=all -v ${pwd} -p 8888 -ti pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime bash

# Now from inside the singularity we create a virtual env to install some additional packages
python -m venv myenv --system-site-packages

# Activate the environment TO BE DONE ALL THE TIME
source myenv/bin/activate
# install packages (to doonly once)
python -m pip install -r requirements.txt

# Make the virtualenv visible to jupyter lab
python -m ipykernel install --user --name=myenv

# Now we can start the jupyter notebook, 
jupyter lab
```

## Datasets
The training dataset is available on CERN EOS to the course students. They are accessible at `/eos/user/d/dvalsecchi/PhDCourse_MLColliderPhysics2024`. 
The dataset is also temporarely publicly available at https://dvalsecc.web.cern.ch/public/datasets/PhDCourse_MLColliderPhysics_2024/training_datasets.tar.gz.

```bash
curl https://dvalsecc.web.cern.ch/public/datasets/PhDCourse_MLColliderPhysics_2024/training_datasets.tar.gz
```



