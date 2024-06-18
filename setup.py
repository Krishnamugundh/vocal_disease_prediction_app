import setuptools

setuptools.setup(
    name='audio_detections',
    version='0.0.2',
    description='Post-installation script for pywin32',
    author='mugu',
    author_email='xyz@gmail.com',
    py_modules=['pywin32_postinstall'],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
    
