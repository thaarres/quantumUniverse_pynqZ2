import os
import tarfile
import shutil
import time
import numpy as np

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def copy_wait_retry(src, dst, sleep=60):
    if not os.path.isfile(src):
        print(f'File {src} not found, waiting {sleep}s before retry')
        time.sleep(sleep)
    shutil.copy(src, dst)

def package(hls_model, X=None, y=None, name=None, sleep_before_retry=60):
    '''Package the hardware build results for HW inference, including test data'''

    odir = hls_model.config.get_output_dir()
    name = hls_model.config.get_project_name()

    if os.path.isdir(f'{odir}/package/'):
        print(f'Found existing package "{odir}/package/", overwriting')
    os.makedirs(f'{odir}/package/', exist_ok=True)
    if not X is None:
        np.save(f'{odir}/package/X.npy', X)
    if not y is None:
        np.save(f'{odir}/package/y.npy', y)

    src = f'{odir}/{name}_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper.bit'
    dst = f'{odir}/package/{name}.bit'
    copy_wait_retry(src, dst)

    src = f'{odir}/{name}_vivado_accelerator/project_1.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh'
    dst = f'{odir}/package/{name}.hwh'
    shutil.copy(src, dst)

    driver = hls_model.config.writer.vivado_accelerator_config.get_driver_file()
    shutil.copy(f'{odir}/{driver}', f'{odir}/package/{driver}')

    make_tarfile(f'{odir}/{name}.tar.gz', f'{odir}/package')