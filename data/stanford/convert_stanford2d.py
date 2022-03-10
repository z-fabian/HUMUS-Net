"""
Converts the original Stanford 2D FSE dataset to a format 
easily accessed by fastMRI modules.

kspace data is scaled by 1e-8 
target is obtained via RSS reconstruction

Stanford 2D FSE dataset: http://www.mridata.org/list?project=Stanford%202D%20FSE
Modified from: https://github.com/MRSRL/mridata-recon/
"""
import os
import numpy as np
import pathlib
import h5py, ismrmrd, tqdm
from argparse import ArgumentParser

try:
    import pyfftw.interfaces.numpy_fft as fft
except:
    from numpy import fft

def ifftnc(x, axes):
    tmp = fft.fftshift(x, axes=axes)
    tmp = fft.ifftn(tmp, axes=axes)
    return fft.ifftshift(tmp, axes=axes)


def fftnc(x, axes):
    tmp = fft.fftshift(x, axes=axes)
    tmp = fft.fftn(tmp, axes=axes)
    return fft.ifftshift(tmp, axes=axes)


def fftc(x, axis=0, do_orthonorm=True):
    if do_orthonorm:
        scale = np.sqrt(x.shape[axis])
    else:
        scale = 1.0
    return fftnc(x, (axis,)) / scale


def ifftc(x, axis=0, do_orthonorm=True):
    if do_orthonorm:
        scale = np.sqrt(x.shape[axis])
    else:
        scale = 1.0
    return ifftnc(x, (axis,)) * scale


def fft2c(x, order='C', do_orthonorm=True):
    if order == 'C':
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[-2:]))
        else:
            scale = 1.0
        return fftnc(x, (-2, -1)) / scale
    else:
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[:2]))
        else:
            scale = 1.0
        return fftnc(x, (0, 1)) / scale


def ifft2c(x, order='C', do_orthonorm=True):
    if order == 'C':
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[-2:]))
        else:
            scale = 1.0
        return ifftnc(x, (-2, -1)) * scale
    else:
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[:2]))
        else:
            scale = 1.0
        return ifftnc(x, (0, 1)) * scale


def fft3c(x, order='C'):
    if order == 'C':
        return fftnc(x, (-3, -2, -1))
    else:
        return fftnc(x, (0, 1, 2))


def ifft3c(x, order='C'):
    if order == 'C':
        return ifftnc(x, (-3, -2, -1))
    else:
        return ifftnc(x, (0, 1, 2))

def _compute_coefficients_ahncho(kscalib):
    """Compute correction coefficients for FSE using Ahn-Cho method.
    kscalib:  [..., channels, segments, kx]
    """
    # offset start and end to avoid effects from fft wrap
    istart = 5
    iend = -5

    kscalib_shape = kscalib.shape
    num_segments = kscalib_shape[-2]
    num_kx = kscalib_shape[-1]

    kscalib = np.reshape(kscalib, [-1, num_segments, num_kx])
    imcalib = ifftc(kscalib, axis=-1)
    imcalib_ref = np.conj(imcalib) * imcalib[:, :1, :]
    p1_calc = np.angle(np.mean(imcalib_ref[:, :, istart:iend]
                               * np.conj(imcalib_ref[:, :, (istart+1):(iend+1)]), axis=-1))
    p1_calc = np.expand_dims(p1_calc, axis=-1)

    x = np.arange(num_kx * 1.0)
    x = np.reshape(x, [1, 1, num_kx])
    imcalib_cor1 = imcalib_ref * np.exp(1j * x * p1_calc)
    p0_calc = np.angle(np.mean(imcalib_cor1, axis=-1))

    p1_calc = np.reshape(p1_calc, kscalib_shape[:-2] + p1_calc.shape[-2:])
    p0_calc = np.reshape(p0_calc, kscalib_shape[:-2] + p0_calc.shape[-1:] + (1,))

    return p0_calc, -p1_calc


def phase_correction(kspace, echo_train, kspace_fse_calib, echo_train_fse_calib):
    """Perform linear phase correction for FSE scans
    kspace: [phases, echoes, slices, channels, kz, ky, kx]
    echo_train: [phases, echoes, slices, 1, segments, 1]
    """
    x = np.arange(kspace.shape[-1] * 1.0)
    x = np.reshape(x, [1, 1, -1])

    p0, p1 = _compute_coefficients_ahncho(kspace_fse_calib)
    ksx = ifftc(kspace, axis=-1)

    num_phases = ksx.shape[0]
    num_echoes = ksx.shape[1]
    num_slices = ksx.shape[2]
    num_kz = ksx.shape[-3]

    for i_phase in range(num_phases):
        for i_echo in range(num_echoes):
            for i_slice in range(num_slices):
                for i_kz in range(num_kz):
                    ind = echo_train[i_phase, i_echo, i_slice, 0, i_kz, :, 0]
                    ind_calib = echo_train_fse_calib[i_phase, i_echo, i_slice, 0, :, 0]
                    ks_slice = ksx[i_phase, i_echo, i_slice, :, i_kz, :, :]

                    p0_slice = p0[i_phase, i_echo, i_slice, :, ind_calib, :]
                    p0_slice = np.transpose(p0_slice[ind, :, :], [1, 0, 2])

                    p1_slice = p1[i_phase, i_echo, i_slice, :, ind_calib, :]
                    p1_slice = np.transpose(p1_slice[ind, :, :], [1, 0, 2])

                    ks_slice *= np.exp(1j * (p0_slice + x * p1_slice))
                    ksx[i_phase, i_echo, i_slice, :, i_kz, :, :] = ks_slice

    kspace_cor = fftc(ksx, axis=-1)

    return kspace_cor

def isrmrmd_user_param_to_dict(header):
    """
    Store ISMRMRD header user parameters in a dictionary.
    Parameter
    ---------
    header : ismrmrd.xsd.ismrmrdHeader
        ISMRMRD header object
    Returns
    -------
    dict
        Dictionary containing custom user parameters
    """
    user_dict = {}
    user_long = list(header.userParameters.userParameterLong)
    user_double = list(header.userParameters.userParameterDouble)
    user_string = list(header.userParameters.userParameterString)
    user_base64 = list(header.userParameters.userParameterBase64)

    for entry in user_long + user_double + user_string + user_base64:
        user_dict[entry.name] = entry.value_

    return user_dict

def load_ismrmrd_to_np(file_name, verbose=False):
    """
    Load data from an ISMRMRD file to a numpy array.
    Raw data from the ISMRMRD file is loaded into a numpy array. If the ISMRMRD file includes the array 'rec_std' that contains the standard deviation of the noise, this information is used to pre-whiten the k-space data. If applicable, a basic phase correction is performed on the loaded k-space data.
    Parameters
    ----------
    file_name : str
        Name of ISMRMRD file
    verbose : bool, optional
        Turn on/off verbose print out
    Returns
    -------
    np.array
        k-space data in an np.array of dimensions [phase, echo, slice, coils, kz, ky, kx]
    ismrmrd.xsd.ismrmrdHeader
        ISMRMRD header object
    """
    dataset = ismrmrd.Dataset(file_name, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    param_dict = isrmrmd_user_param_to_dict(header)

    num_kx = header.encoding[0].encodedSpace.matrixSize.x
    num_ky = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1
    num_kz = header.encoding[0].encodingLimits.kspace_encoding_step_2.maximum + 1
    num_channels = header.acquisitionSystemInformation.receiverChannels
    num_slices = header.encoding[0].encodingLimits.slice.maximum + 1
    num_echoes = header.encoding[0].encodingLimits.contrast.maximum + 1
    num_phases = header.encoding[0].encodingLimits.phase.maximum + 1
    num_segments = header.encoding[0].encodingLimits.segment.maximum + 1
    is_fse_with_calib = num_segments > 1

    chop_y = 1 - int(param_dict.get('ChopY', 1))
    chop_z = 1 - int(param_dict.get('ChopZ', 1))

    try:
        rec_std = dataset.read_array('rec_std', 0)
        rec_weight = 1.0 / (rec_std ** 2)
        rec_weight = np.sqrt(rec_weight / np.sum(rec_weight))
    except Exception:
        rec_weight = np.ones(num_channels)
    opt_mat = np.diag(rec_weight)

    if verbose:
        print("Data dims: (%d, %d, %d, %d, %d, %d, %d)" % (num_kx, num_ky, num_kz,
                                                           num_channels, num_slices,
                                                           num_echoes, num_phases))
    kspace = np.zeros([num_phases, num_echoes, num_slices, num_channels,
                       num_kz, num_ky, num_kx], dtype=np.complex64)

    if is_fse_with_calib:
        echo_train = np.zeros([num_phases, num_echoes, num_slices, 1, num_kz, num_ky, 1],
                              dtype=np.uint)
        kspace_fse_cal = np.zeros([num_phases, num_echoes, num_slices, num_channels,
                                   num_segments, num_kx], dtype=np.complex64)
        echo_train_fse_cal = np.zeros([num_phases, num_echoes, num_slices, 1, num_segments, 1],
                                      dtype=np.uint)

    max_slice = 0
    wrap = lambda x: x
    if verbose:
        print("Loading data...")
        wrap = tqdm
    try:
        num_acq = dataset.number_of_acquisitions()
    except:
        print("Unable to determine number of acquisitions! Empty?")
        return
    for i in wrap(range(num_acq)):
        acq = dataset.read_acquisition(i)
        i_ky = acq.idx.kspace_encode_step_1 # pylint: disable=E1101
        i_kz = acq.idx.kspace_encode_step_2 # pylint: disable=E1101
        i_echo = acq.idx.contrast           # pylint: disable=E1101
        i_phase = acq.idx.phase             # pylint: disable=E1101
        i_slice = acq.idx.slice             # pylint: disable=E1101
        if i_slice > max_slice:
            max_slice = i_slice
        sign = (-1) ** (i_ky * chop_y + i_kz * chop_z)
        data = np.matmul(opt_mat.T, acq.data) * sign
        if i_kz < num_kz:
            i_segment = acq.idx.segment # pylint: disable=E1101
            if i_ky < num_ky:
                kspace[i_phase, i_echo, i_slice, :, i_kz, i_ky, :] = data
                if is_fse_with_calib:
                    echo_train[i_phase, i_echo, i_slice, 0, i_kz, i_ky, 0] = i_segment
            elif is_fse_with_calib:
                kspace_fse_cal[i_phase, i_echo, i_slice, :, i_ky - num_ky, :] = data
                echo_train_fse_cal[i_phase, i_echo, i_slice, 0, i_ky - num_ky, 0] = i_segment
    dataset.close()

    max_slice += 1
    if num_slices != max_slice:
        if verbose:
            print("Actual number of slices different: %d/%d" % (max_slice, num_slices))
        kspace = kspace[:, :, :max_slice, :, :, :, :]
        if is_fse_with_calib:
            echo_train = echo_train[:, :, :max_slice, :, :, :, :]
            kspace_fse_cal = kspace_fse_cal[:, :, :max_slice, :, :, :]
            echo_train_fse_cal = echo_train_fse_cal[:, :, :max_slice, :, :, :]

    if is_fse_with_calib:
        if verbose:
            print("FSE phase correction...")
        if 0:
            print("writing files for debugging...")
            cfl.write("kspace", kspace)
            cfl.write("echo_train", echo_train)
            cfl.write("kspace_fse_cal", kspace_fse_cal)
            cfl.write("echo_train_fse_cal", echo_train_fse_cal)
        kspace_cor = phase_correction(kspace, echo_train, kspace_fse_cal, echo_train_fse_cal)
        # for debugging
        if 0:
            cfl.write("kspace_orig" , kspace)
        kspace = kspace_cor

    return kspace, header

# Functions to generate reconstruction target
def ifft2_np(x):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x.astype(np.complex64), axes=[-2, -1]), norm='ortho'), axes=[-2, -1]).astype(np.complex64)

def kspace_to_target(x):
    return np.sqrt(np.sum(np.square(np.abs(ifft2_np(x))), axis=-3)).astype(np.float32)

# Main conversion code
def cli_main(args):
    print(args)
    if not os.path.exists(args.output_dir):
        print('Creating output directory...')
        os.makedirs(args.output_dir)
    
    mri_files = list(pathlib.Path(args.input_dir).glob('*.h5'))
    print(mri_files)
    for i, mri_f in zip(range(len(mri_files)), sorted(mri_files)):
        kspace, header = load_ismrmrd_to_np(mri_f, verbose=False)
        print('converting ', i+1, '/', len(mri_files))
        scaling = 1e8  # scale measurements to similar range as fastMRI
        kspace = kspace[0, 0, :, :, 0, :, :] / scaling
        target = kspace_to_target(kspace)
        save_file = os.path.join(args.output_dir, str(mri_f.name))
        data = h5py.File(save_file, 'w')
        data.create_dataset('kspace', data=kspace)
        data.create_dataset('reconstruction_rss', data=target)
        data.attrs.create('max', data=target.max())
        data.close()
    print('Finished.')
    
def build_args():
    parser = ArgumentParser()

    # client arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory where the original Stanford2D dataset is located.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save the converted dataset.",
    )
    
    args = parser.parse_args()
    return args
    
def run_cli():
    args = build_args()
    # run conversion
    cli_main(args)

if __name__ == "__main__":
    run_cli()