#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from setuptools import setup, Extension
from collections import defaultdict
from distutils import core
from distutils.core import Distribution
from distutils.errors import DistutilsArgError
import setuptools.command.build_ext
import setuptools.command.install
import distutils.command.clean
import distutils.sysconfig

import fnmatch
import os
import platform
import sys

def printValues():
	vars = distutils.sysconfig.get_config_vars('CC', 'CXX', 'OPT', 'BASECFLAGS', 'CCSHARED', 'LDSHARED', 'SO')
	for i in range(len(vars)):
		if vars[i] is None:
			vars[i] = ""
	(cc, cxx, opt, basecflags, ccshared, ldshared, so_ext) = vars
	print(vars)


def get_libcarla_extensions():
	printValues()
	include_dirs = ['dependencies/include']

	library_dirs = ['dependencies/lib']
	libraries = []


	sources = ['source/libcarla/libcarla.cpp'] #['boostPython.cpp'] # 

	def walk(folder, file_filter='*'):
		for root, _, filenames in os.walk(folder):
			for filename in fnmatch.filter(filenames, file_filter):
				yield os.path.join(root, filename)

	if os.name == "posix":
		# @todo Replace deprecated method.
		linux_distro = platform.dist()[0]  # pylint: disable=W1505
		if linux_distro.lower() in ["ubuntu", "debian", "deepin"]:
			pwd = os.path.dirname(os.path.realpath(__file__))
			pylib = "libboost_python%d%d.a" % (sys.version_info.major,
											   sys.version_info.minor)
			extra_link_args = [
				os.path.join(pwd, 'dependencies/lib/libcarla_client.a'),
				os.path.join(pwd, 'dependencies/lib/librpc.a'),
				os.path.join(pwd, 'dependencies/lib/libboost_filesystem.a'),
				os.path.join(pwd, 'dependencies/lib/libRecast.a'),
				os.path.join(pwd, 'dependencies/lib/libDetour.a'),
				os.path.join(pwd, 'dependencies/lib/libDetourCrowd.a'),
				os.path.join(pwd, 'dependencies/lib', pylib)]
			extra_compile_args = [
				'-isystem', 'dependencies/include/system', '-fPIC', '-std=c++14',
				'-Werror', '-Wall', '-Wextra', '-Wpedantic', '-Wno-self-assign-overloaded',
				'-Wdeprecated', '-Wno-shadow', '-Wuninitialized', '-Wunreachable-code',
				'-Wpessimizing-move', '-Wold-style-cast', '-Wnull-dereference',
				'-Wduplicate-enum', '-Wnon-virtual-dtor', '-Wheader-hygiene',
				'-Wconversion', '-Wfloat-overflow-conversion',
				'-DBOOST_ERROR_CODE_HEADER_ONLY', '-DLIBCARLA_WITH_PYTHON_SUPPORT'
			]
			if 'BUILD_RSS_VARIANT' in os.environ and os.environ['BUILD_RSS_VARIANT'] == 'true':
				print('Building AD RSS variant.')
				extra_compile_args += ['-DLIBCARLA_RSS_ENABLED']
				extra_link_args += [os.path.join(pwd, 'dependencies/lib/libad-rss.a')]

			if 'TRAVIS' in os.environ and os.environ['TRAVIS'] == 'true':
				print('Travis CI build detected: disabling PNG support.')
				extra_link_args += ['-ljpeg', '-ltiff']
				extra_compile_args += ['-DLIBCARLA_IMAGE_WITH_PNG_SUPPORT=false']
			else:
				extra_link_args += ['-lpng', '-ljpeg', '-ltiff']
				extra_compile_args += ['-DLIBCARLA_IMAGE_WITH_PNG_SUPPORT=true']
			# @todo Why would we need this?
			include_dirs += ['/usr/lib/gcc/x86_64-linux-gnu/7/include']
			library_dirs += ['/usr/lib/gcc/x86_64-linux-gnu/7']
			extra_link_args += ['/usr/lib/gcc/x86_64-linux-gnu/7/libstdc++.a']
		else:
			raise NotImplementedError
	elif os.name == "nt":
		sources += [x for x in walk('dependencies/include/carla', '*.cpp')]

		pwd = os.path.dirname(os.path.realpath(__file__))
		#pylib = 'libboost_python%d%d' % (
		#	sys.version_info.major,
		#	sys.version_info.minor)

		extra_link_args = ['/NODEFAULTLIB:LIBCMT.LIB', 'shlwapi.lib']


		"""
		required_libs = [
			#pylib, 
			'libboost_filesystem-vc141-mt-x64-1_72.lib', 'libboost_python37-vc141-mt-x64-1_72.lib', 
			'libboost_system-vc141-mt-x64-1_72.lib', 'libboost_date_time-vc141-mt-x64-1_72.lib',
			'rpc.lib', 'carla_client.lib',
			'libpng.lib', 'zlib.lib',
			'Recast.lib', 'Detour.lib', 'DetourCrowd.lib']
		"""
		required_libs = ['rpc.lib', 'libboost_filesystem-vc141-mt-x64-1_72.lib', 'libboost_python37-vc141-mt-x64-1_72.lib', 'libboost_system-vc141-mt-x64-1_72.lib', 
							'libboost_date_time-vc141-mt-x64-1_72.lib', 
			'carla_client.lib',
			'libpng.lib', 'zlib.lib',
			'Recast.lib', 'Detour.lib', 'DetourCrowd.lib']

		# Search for files in 'PythonAPI\carla\dependencies\lib' that contains
		# the names listed in required_libs in it's file name
		libs = [x for x in os.listdir('dependencies/lib') if any(d in x for d in required_libs)]

		print("###### Adding libs: ", libs)

		for lib in libs:
			extra_link_args.append(os.path.join(pwd, 'dependencies/lib', lib))

		print("#### Extra link args: ", extra_link_args)

		# https://docs.microsoft.com/es-es/cpp/porting/modifying-winver-and-win32-winnt
		extra_compile_args = ['/MD', '/Z7',
							  '/EHa', '/DNOMINMAX',
							  '/wd4267', '/wd4251', '/wd4522', '/wd4522', '/wd4838',
							  '/wd4305', '/wd4244', '/wd4190', '/wd4101', '/wd4996',
							  '/wd4275',
			'/experimental:external', '/external:I', 'dependencies/include/system',
			'/DBOOST_ALL_NO_LIB', '/DBOOST_PYTHON_STATIC_LIB',
			'/DBOOST_ERROR_CODE_HEADER_ONLY', '/D_WIN32_WINNT=0x0600', '/DHAVE_SNPRINTF',
			'/DLIBCARLA_WITH_PYTHON_SUPPORT', '-DLIBCARLA_IMAGE_WITH_PNG_SUPPORT=true']


		print("#### Extra compile args: ", extra_compile_args)

	else:
		raise NotImplementedError

	depends = [x for x in walk('source/libcarla')]
	depends += [x for x in walk('dependencies')]

	#print("#### Depends", depends)

	def make_extension(name, sources):

		return Extension(
			name,
			sources=sources,
			include_dirs=include_dirs,
			library_dirs=library_dirs,
			libraries=libraries,
			extra_compile_args=extra_compile_args,
			extra_link_args=extra_link_args,
			language='c++14',
			depends=depends)

	print('compiling:\n  - %s' % '\n  - '.join(sources))

	return [make_extension('carla.libcarla', sources)]

 

setup(
	name='carla',
	version='0.9.8',
	package_dir={'': 'source'},
	packages=['carla'],
	ext_modules=get_libcarla_extensions(),
	license='MIT License',
	description='Python API for communicating with the CARLA server.',
	url='https://github.com/carla-simulator/carla',
	author='The CARLA team',
	author_email='carla.simulator@gmail.com',
	include_package_data=True)
