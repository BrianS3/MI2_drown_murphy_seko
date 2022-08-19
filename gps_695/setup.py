from setuptools import setup

setup(name='gps_695',
      version='0.1',
      description='University of Michigan Milestone 2 Project',
      url='https://github.com/BrianS3/MI2_drown_murphy_seko',
      author='Drown, Gretchyn; Murphy, Patrick; Seko, Brian',
      author_email='bseko@umich.edu',
      license='MIT',
      packages=['gps_695'],
      install_requires=[
            'pip',
            'json',
            'dotenv',
            'requests'
      ],
      zip_safe=False)