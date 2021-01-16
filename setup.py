import setuptools

requirements = [
    "tensorflow>=2.4.0",
    "tensorflow_io>=0.17",
    "numpy>=1.19.2,<1.20.0",
    "soundfile>=0.10.3",
    "PyYAML>=5.3.1",
]

setuptools.setup(
    name="vistec-ser",
    version="0.2.1a1",
    author="Chompakorn Chaksangchaichot",
    author_email="chompakorn.cc@gmail.com",
    description="Speech Emotion Recognition models and training using Tensorflow 2.x",
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tann9949/vistec-ser',
    packages=setuptools.find_packages(include=['vistec_ser*']),
    install_requires=requirements,
    classifiers=[
        # "2 - Pre-Alpha", "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.6'
)
