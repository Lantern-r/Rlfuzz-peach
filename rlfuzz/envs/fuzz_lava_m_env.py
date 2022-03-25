import rlfuzz
from rlfuzz.envs.fuzz_base_env import FuzzBaseEnv

import os


class FuzzBase64Env(FuzzBaseEnv):
    def __init__(self):
        self._target_path = rlfuzz.base64_target_path()
        self._args = ['-d']
        self._seed = b''  # 指定初始变异的文件
        self._input_maxsize = 32 * 1024  # 最大输入文件的大小
        super(FuzzBase64Env, self).__init__()

    def set_seed(self, seed):
        assert len(seed) > 0
        assert isinstance(seed, bytes)
        self._seed = seed
        self._input_maxsize = len(seed)
        self.reset()


class FuzzMd5sumEnv(FuzzBaseEnv):
    def __init__(self):
        self._target_path = rlfuzz.md5sum_target_path()
        self._args = ['-c']
        self._seed = b''  # 指定初始变异的文件
        self._input_maxsize = 32 * 1024  # 最大输入文件的大小
        super(FuzzMd5sumEnv, self).__init__()

    def set_seed(self, seed):
        assert len(seed) > 0
        assert isinstance(seed, bytes)
        self._seed = seed
        self._input_maxsize = len(seed)
        self.reset()


class FuzzUniqEnv(FuzzBaseEnv):
    def __init__(self):
        self._target_path = rlfuzz.uniq_target_path()
        self._args = []
        self._seed = b''  # 指定初始变异的文件
        self._input_maxsize = 32 * 1024  # 最大输入文件的大小
        super(FuzzUniqEnv, self).__init__()

    def set_seed(self, seed):
        assert len(seed) > 0
        assert isinstance(seed, bytes)
        self._seed = seed
        self._input_maxsize = len(seed)
        self.reset()


class FuzzWhoEnv(FuzzBaseEnv):
    def __init__(self):
        self._target_path = rlfuzz.who_target_path()
        self._args = []
        self._seed = b''  # 指定初始变异的文件
        self._input_maxsize = 32 * 1024  # 最大输入文件的大小
        super(FuzzWhoEnv, self).__init__()

    def set_seed(self, seed):
        assert len(seed) > 0
        assert isinstance(seed, bytes)
        self._seed = seed
        self._input_maxsize = len(seed)
        self.reset()


class FuzzgzipEnv(FuzzBaseEnv):
    def __init__(self):
        self._target_path = rlfuzz.gzip_target_path()
        self._args = ['-d']
        self._seed = b''  # 指定初始变异的文件
        self._suffix = 'afl_out.gz'
        self._input_maxsize = 32 * 1024  # 最大输入文件的大小
        self.peachflag = True
        if self.peachflag:
            self._Seed_Path = '/home/real/rlfuzz-socket/rlfuzz/mods/gzip-mod/seed/1.ppt.gz'
            self._dataModelName = 'gzip_file'
            self._PitPath = 'file:test/pit/GZIP_DataModel.xml'
        super(FuzzgzipEnv, self).__init__(PeachFlag=self.peachflag)

    def set_seed(self, seed):
        assert len(seed) > 0
        assert isinstance(seed, bytes)
        self._seed = seed
        self._input_maxsize = len(seed) * 10
        self.reset()


class FuzzpngquantEnv(FuzzBaseEnv):
    def __init__(self):
        self._target_path = rlfuzz.pngquant_target_path()
        self._args = ['-f']
        self._seed = b''  # 指定初始变异的文件
        self._suffix = 'afl_out.png'
        self._input_maxsize = 32 * 1024  # 最大输入文件的大小
        self.peachflag = True
        if self.peachflag:
            self._Seed_Path = '/home/real/rlfuzz-socket/rlfuzz/mods/pngquant-mod/pngquant-master/test/img/metadata.png'
            self._dataModelName = 'PNG'
            self._PitPath = 'file:test/pit/png_datamodel.xml'
        super(FuzzpngquantEnv, self).__init__(PeachFlag=self.peachflag)

    def set_seed(self, seed):
        assert len(seed) > 0
        assert isinstance(seed, bytes)
        self._seed = seed
        self._input_maxsize = len(seed) * 10
        self.reset()
