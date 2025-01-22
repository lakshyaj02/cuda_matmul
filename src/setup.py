from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

'''
Use the following to build extension:
python3 setup.py install
'''

setup(
    name='custom_mm',
    ext_modules=[
        CUDAExtension('custom_mm', [
            'custom_mm.cpp',
            'baseline_mm.cu',
            'sparse_mm.cu',
            'naive_sparse_mm.cu'
        ],
        extra_compile_args={'nvcc': ['-lcusparse']})
    ],
    packages=["blocksparse"],
    package_dir={
        "": ".",
        "blocksparse": "./matmul"
    },
    cmdclass={
        'build_ext': BuildExtension
    }
    )