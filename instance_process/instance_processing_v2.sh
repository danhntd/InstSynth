# Define dataset paths
conda activate cityscapes
cd /mmlabworkspace/WorkSpaces/danhnt/InstMask2Image/cityscapes-process-official

DATASET_PATH="../cityscapes_synthesis_temp/"                           # Path to cityscape dataset download from web
INPAINTING_PATH="../cityscapes_synthesis/DiffPainting/version01"       # Path to inpainted image from cityscape dataset
OUTPUT_DATASET_PATH="cityscapes_diffpainting"                          # Path to new dataset

# Step 1: Instance selection
# k :  number of instance to be selected 
python process_clean_ver.py \
  --k 3 \
  --train_folder "$DATASET_PATH"

# Step 2: Integrate inpainted images with metadata and cropped region coordinates 
python inpainted_integration.py \
  --inpainting_dir "$INPAINTING_PATH" \
  --metadata_path "${DATASET_PATH}metadata.json" \
  --cropped_region_coordinates "${DATASET_PATH}cropped_region_coordinates.json" \
  --output_folder_name "$OUTPUT_DATASET_PATH"

# Step 3: Move Raw Data to the Output Dataset Folder
python move_folder_clean_ver.py \
  --src_path "$DATASET_PATH" \
  --des_path "$OUTPUT_DATASET_PATH"

# Step 4: Crop raw color & labelIds in the raw city folder to match new image pairs
python drop_color_and_labelIds_clean_ver.py \
  --cropped_region_coordinates_path "${OUTPUT_DATASET_PATH}/cropped_region_coordinates.json"

# Step 5: Calculate Polygons for Inpainted Images
cd cityscapes-to-coco-conversion
python main_defined_clean_ver.py \
  --dataset cityscapes \
  --datadir "../$OUTPUT_DATASET_PATH"


