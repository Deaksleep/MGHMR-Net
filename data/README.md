# External dataset

The dataset is intentionally not distributed with this code repository. Obtain
the image archive and label workbook from the project's separate data record.
After download, either place them in the default local layout below or pass
their external paths to the training and validation commands.

```text
data/
├── images/          # extracted .tif/.tiff files
└── labels.xlsx      # measurement table
```

All common data formats under `data/` are ignored by Git to prevent accidental
publication. To extract a downloaded archive into the default directory:

```bash
python scripts/prepare_data.py --archive /path/to/images.zip
python scripts/validate_data.py
```

## Label schema

| Column | Meaning | Training filter |
| --- | --- | --- |
| `filename` | TIFF basename used to join image and measurement records | Must exist in the extracted image set |
| `modulus` | Positive regression target (elastic modulus) | Must be non-null and greater than zero |
| `R` | Measurement fit/correlation quality | Must be greater than `--r-threshold` (default: 0.75) |
| `day` | Culture day | Must be one of 1, 4, 7, 10, or 13 |

Days are grouped as `Day_1={1}`, `Day_4_7={4,7}`, and
`Day_10_13={10,13}`. The label workbook contains additional records; only rows
whose filenames are included in the selected image archive enter the analysis.


