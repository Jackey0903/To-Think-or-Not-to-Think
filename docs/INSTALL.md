# Installation

The project uses two environments:

- `think`: Ref-Thinker training and generation.
- `dino`: GroundingDINO and SAM2 based grounding and segmentation.

## Ref-Thinker Environment

```bash
conda env create -f think_environment.yml
conda activate think
```

## Ground-Segment Environment

```bash
conda env create -f dinosam2_environment.yml
conda activate dino
```

Install Grounded-SAM-2 under `ground_segment_scripts/`:

```bash
cd ground_segment_scripts
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
# Follow the upstream installation instructions for Grounded-SAM-2.
```

For compatibility with the scripts in this repository, expose these paths from the repository root:

```bash
ln -s ground_segment_scripts/Grounded-SAM-2/grounding_dino grounding_dino
ln -s ground_segment_scripts/Grounded-SAM-2/sam2 sam2
ln -s ground_segment_scripts/Grounded-SAM-2/checkpoints checkpoints
ln -s ground_segment_scripts/Grounded-SAM-2/gdino_checkpoints gdino_checkpoints
```

Then download the GroundingDINO and SAM2 weights using the upstream scripts:

```bash
bash ground_segment_scripts/Grounded-SAM-2/gdino_checkpoints/download_ckpts.sh
bash ground_segment_scripts/Grounded-SAM-2/checkpoints/download_ckpts.sh
```

## Check

```bash
python scripts/check_smoke.py
```

