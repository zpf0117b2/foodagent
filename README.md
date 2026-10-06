# FoodAgent

Code for *FoodAgent: A Multi-modal Mixture of Experts Reasoning Agent for Divide-and-Conquer Food Nutrition
Estimation* (IEEE BSN 2025), using **Qwen2.5-VL** as the reasoning agent and
[MFP3D](https://arxiv.org/abs/2411.10492) as the volume expert.

FoodAgent works on one food photo:

1. It splits the dish into food elements.
2. It sends each element to one of three experts:
   - main and visually clear → MFP3D monocular volume estimation;
   - main but not clear → retrieval from a nutrition database;
   - minor → direct Qwen2.5-VL estimation.
3. It sums the experts' results into the dish's calories, mass, fat, carbs and protein.

Everything runs locally. No API keys or online services are needed at run time.

```
foodagent/            library code
  agent.py            element analysis + three-category routing (Qwen2.5-VL)
  experts/            volume expert (SAM + ZoeDepth + MFP3D), retrieval expert, direct-LLM expert
  generator.py        response generator (mass = volume x density, unit counting, aggregation)
  pipeline.py         FoodAgent and the Qwen2.5-VL direct baseline
  mfp3d/              MFP3D: point-cloud reconstruction, CurveNet + ResNet50 model, MetaFood3D tools
  data/nutrition5k.py Nutrition5k loader
scripts/              data preparation, training, evaluation, demo
```

## 1. Environment

The code needs Python 3.10+, PyTorch with CUDA, and a GPU. Qwen2.5-VL-7B in bf16 needs about 17 GB of GPU memory;
with `--quantization 4bit` (install `bitsandbytes`) it needs about 6 GB.

```bash
conda create -n foodagent python=3.11 -y
conda activate foodagent
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA version
pip install -r requirements.txt
```

The pretrained models download automatically from the Hugging Face Hub on first use:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `facebook/sam-vit-huge`
- `Intel/zoedepth-nyu-kitti`
- the torchvision ResNet50 ImageNet weights

To run offline, download them once and set `HF_HUB_OFFLINE=1`.

## 2. Datasets: access and placement

All data goes under `data/` in the repository root:

```
data/
├── nutrition5k/                       # evaluation (FoodAgent)
│   ├── metadata/dish_metadata_cafe1.csv
│   ├── metadata/dish_metadata_cafe2.csv
│   ├── metadata/ingredients_metadata.csv
│   ├── dish_ids/splits/{rgb,depth}_{train,test}_ids.txt
│   └── imagery/realsense_overhead/<dish_id>/rgb.png
├── fdc/                               # USDA FoodData Central CSVs (nutrition reference table)
│   └── FoodData_Central_sr_legacy_food_csv_<date>/{food,nutrient,food_nutrient,food_portion,measure_unit}.csv
├── nutrition_db.csv                   # built in step 3
└── MetaFood3D/                        # MFP3D training (extracted archives, see 2.3)
```

### 2.1 Nutrition5k (evaluation)

Source: <https://github.com/google-research-datasets/Nutrition5k>. It is a public Google Cloud bucket
(`gs://nutrition5k_dataset/nutrition5k_dataset/`), released under CC BY 4.0.

Only the metadata, the split lists, and the overhead RGB images are needed, about 200 MB for the test split instead
of the full 181 GB:

```bash
python scripts/fetch_nutrition5k_subset.py --out data/nutrition5k --splits depth_test
```

The `gsutil` equivalent is `gsutil -m cp -r gs://nutrition5k_dataset/nutrition5k_dataset/{metadata,dish_ids} data/nutrition5k/`,
plus `imagery/realsense_overhead/<dish_id>/rgb.png` for each id in `dish_ids/splits/depth_test_ids.txt`.

**Test set.** Evaluation uses `depth_test_ids.txt`, which lists the dishes that have the overhead camera image. Only
`rgb.png` is used as input; depth is never read.

### 2.2 USDA FoodData Central (nutrition reference table)

The retrieval expert looks foods up in an offline table of reference servings ("1 cup = 128 g, 206 kcal, ..."). Build
it from USDA FoodData Central, which is public domain:

1. Go to <https://fdc.nal.usda.gov/download-datasets/>.
2. Download the **SR Legacy** CSV. Optionally also download **FNDDS** for more prepared dishes.
3. Unzip into `data/fdc/`.

### 2.3 MetaFood3D (MFP3D training)

Source: <https://lorenz.ecn.purdue.edu/~food3d/>, CC BY-NC 4.0. Access needs a request form; after approval you get
download access.

- **Archives to download:** `3D_Mesh`, `RGBD_videos`, `Nutrition values`, and the `Readme`.
- **Placement:** extract them under `data/MetaFood3D/`.
- **Contents per object:** a textured scan, RGB-D video frames with masks and depth, weight, and
  energy / protein / carb / fat.

**Alternative.** The MFP3D authors share preprocessed train/test H5 files, which already contain point clouds,
images, and targets; the link is in <https://github.com/jingema99/MFP3D>. Place them in
`data/mfp3d_release/{gt,depth,tripoSR}/{train,test}.h5`. They load directly in step 4b. They use the authors'
point-cloud convention (`--recon_mode uvz`), so they are meant for reproducing MFP3D's own tables. For use inside
FoodAgent, build the data with step 4a.

## 3. Build the nutrition reference table

```bash
python scripts/build_local_db.py --from_fdc data/fdc --out data/nutrition_db.csv
```

Each food produces:

- a `100 g` reference row;
- one row per household portion, used for unit counting.

## 4. Train MFP3D (volume expert)

### 4a. From raw MetaFood3D

1. Inspect the extracted layout, then choose the glob patterns for frames, masks, depth, meshes, and the nutrition
   file accordingly:

   ```bash
   python scripts/mfp3d_build_dataset.py --inspect data/MetaFood3D
   ```

2. Build the manifest, with one row per object. The `<...>` placeholders come from the inspect output.

   ```bash
   python scripts/mfp3d_build_dataset.py manifest \
       --objects_glob "data/MetaFood3D/<video folder>/*" \
       --frame_pattern "**/<rgb frames>/*.jpg" --mask_pattern "**/<masks>/*.png" --depth_pattern "**/<depth>/*.png" \
       --mesh_pattern "**/*.obj" \
       --nutrition_csv "data/MetaFood3D/<nutrition file>.csv" --id_col <object id column> \
       --colmap weight=<col> energy=<col> protein=<col> fat=<col> carb=<col> \
       --frames_per_object 4 --out data/metafood3d/manifest.csv
   ```

3. Reconstruct point clouds and crops, split by object (510 train / 127 test), and write the H5 files:

   ```bash
   python scripts/mfp3d_build_dataset.py build --manifest data/metafood3d/manifest.csv \
       --out_dir data/metafood3d/depth_metric --cloud depth --mode metric --anchor sensor \
       --mesh_scale_cm 100 --sensor_depth_to_cm 0.1
   ```

   - **`--mesh_scale_cm`** converts mesh units to cm (metres → 100, mm → 0.1).
   - **`--sensor_depth_to_cm`** converts depth-frame units to cm (mm → 0.1).
   - **Volume labels** are computed from the closed scanned mesh when the nutrition file has none.
   - **Other point-cloud variants:** `--cloud gt` uses mesh surface samples, `--cloud gt_normalized` uses unit-cube
     normalised samples, and `--cloud provided` uses a point-cloud file named in the manifest.

### 4b. Train and test

```bash
python scripts/mfp3d_train.py --data_dir data/metafood3d/depth_metric --target volume
python scripts/mfp3d_test.py --checkpoint outputs/mfp3d/depth_metric/volume/best.pt \
    --h5 data/metafood3d/depth_metric/test.h5
```

- **Training settings:** AdamW, lr 1e-3, batch 16, 200 epochs, L1 loss, 1,024 points. The best checkpoint is chosen
  on a validation split taken from the training objects.
- **Other targets:** `--target energy|weight|protein|fat|carb`.
- **Modality ablations:** point-cloud only with `--no_image`, image only with `--point_encoder none`.
- **Authors' H5 files:** `--data_dir data/mfp3d_release/depth --recon_mode uvz`.

The checkpoint stores its reconstruction settings, and FoodAgent rebuilds point clouds the same way at inference. On
Nutrition5k, the ZoeDepth scale is anchored to the 35.9 cm camera height of the capture rig.

## 5. Run FoodAgent on Nutrition5k

```bash
CKPT=outputs/mfp3d/depth_metric/volume/best.pt
python scripts/run_eval.py --data_root data/nutrition5k --task total --method foodagent --mfp3d_ckpt $CKPT
```

| Experiment | Arguments |
|---|---|
| FoodAgent, dish nutrition (Table I) | `--task total --method foodagent --mfp3d_ckpt $CKPT` |
| Ablation: without MFP3D | `--task total --method foodagent --ablation no_mfp3d` |
| Ablation: without nutrition database | `--task total --method foodagent --ablation no_nutritionx --mfp3d_ckpt $CKPT` |
| Qwen2.5-VL direct baseline | `--task total --method qwen_direct` |
| Referring nutrition, 1st/2nd caloric ingredient (Table II) | the same commands with `--task refer` |

Useful options:

- **`--quantization 4bit`** for GPUs with less than 24 GB.
- **`--limit N`** to run on the first N dishes.
- **`--model Qwen/Qwen2.5-VL-3B-Instruct`** for the smaller model.
- **`--local_db_csv`** if the nutrition table is not at `data/nutrition_db.csv`.

Runs resume after interruption, because model responses are cached in `.cache/`.

Outputs go to `results/<task>_<method>_<split>/`:

- `predictions.jsonl`: per-dish trace with elements, routing, volumes, references, and raw model outputs.
- `metrics.json`: MAE and PMAE (`100 · MAE / mean(ground truth)`) for calories, mass, fat, carb, and protein.
- For `--task total`, `predictions.csv` / `groundtruth.csv` in the format of Nutrition5k's official
  `scripts/compute_eval_statistics.py`.

### Single image

```bash
python scripts/demo.py path/to/dish.png --mfp3d_ckpt $CKPT
python scripts/demo.py path/to/dish.png --mfp3d_ckpt $CKPT --refer "scrambled eggs" "bacon"
```

## Citation

```bibtex
@inproceedings{zhang2025foodagent,
  title     = {FoodAgent: A Multi-modal Mixture of Experts Reasoning Agent for Divide-and-Conquer Food Nutrition Estimation},
  author    = {Zhang, Pengfei and Song, Yutong and Lyu, Chenhan and Wang, Ziyu and Rahmani, Amir M.},
  booktitle = {IEEE International Conference on Body Sensor Networks (BSN)},
  year      = {2025}
}
```
