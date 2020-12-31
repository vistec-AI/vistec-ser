import setuptools

requirements = [
    "tensorflow>=2.4.0",
    "tensorflow_io>=0.17",
    "librosa>=0.8.0",
    "nlpaug>=1.1.1",
]

setuptools.setup(
    name="VistecSER",
    version="0.1a",
    author="Chompakorn Chaksangchaichot",
    author_email="chompakorn.cc@gmail.com",
    description="Speech Emotion Recognition models and training using Tensorflow 2.x",
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tann9949/VistecSER',
    packages=setuptools.find_packages(include=['vistec_ser*']),
    install_requires=requirements,
    python_requires='>=3.8'
)