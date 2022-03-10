## Downloading and converting the Stanford MRI datasets
Follow these instructions to batch download the **Stanford 2D FSE** and **Stanford Fullysampled 3D FSE Knees** datasets. These datasets can be found [here](http://mridata.org/), however batch download is currently not supported on the website.

Here I describe how to download the datasets using the `mridata` Python command line tool. The MRAugment repository uses the fastMRI codebase to handle MRI datasets, therefore we need to convert the downloaded datasets to have a format similar to fastMRI data. This is done through tools from [mridata-recon](https://github.com/MRSRL/mridata-recon/) repository.

The size of the datasets is shown in the following table.
 |                     | Original | Converted |
 | ------------------- | -------- | --------- |
 | **Stanford 2D FSE** |    40G   |    37G    |
 | **Stanford 3D FSE** |    32G   |    34G    |

# Requirements
- Python3

# Instructions
1. In order to install `mridata` first you need to install `requests`, `tqdm`, and `boto` using

   `pip install requests boto3 tqdm`

2. Next, install `mridata` using

   `pip install mridata`

3. Download the dataset. Navigate to the directory where you want to download the dataset and make sure that the correct UUID text file ([*Stanford_2D_TSE_uuid.txt*](Stanford_2D_TSE_uuid.txt) or [*Stanford_3D_FSE_knee_uuid.txt*](Stanford_3D_FSE_knee_uuid.txt)) is saved in the same folder. You can find both of these text files in this repository. To dowload the datasets run the command

   `mridata batch_download Stanford_2D_FSE_uuid.txt`

   or

   `mridata batch_download Stanford_3D_FSE_knee_uuid.txt`

   depending on which dataset you would like to download.
   
4. Convert the datasets using the scripts in this repository. Run
    `python convert_stanford2d.py --input_dir --output_dir`
    
    or 
    
    `python convert_stanford3d.py --input_dir --output_dir`
    
    and replace the input and output directories with the desired path on your machine. 

# References
Both of these datasets can be downloaded directly from http://mridata.org/
- **Stanford 2D FSE**: Joseph Y. Cheng, https://github.com/MRSRL/mridata-recon/
- **Stanford Fullysampled 3D FSE Knees**: Epperson K, Sawyer AM, Lustig M, Alley M, Uecker M., *Creation Of Fully Sampled MR Data Repository For Compressed Sensing Of The Knee. In: Proceedings of the 22nd Annual Meeting for Section for Magnetic Resonance Technologists, 2013*
- `mridata` command line tool repository: https://github.com/mikgroup/mridata-python
- Tools used to extract kspace data from original datasets: https://github.com/MRSRL/mridata-recon/