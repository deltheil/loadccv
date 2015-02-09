package = "loadccv"
version = "scm-1"

source = {
   url = "git://github.com/deltheil/loadccv.git",
}

description = {
   summary = "Load ccv networks in Torch7",
   detailed = [[
      Load a ccv convolutional neural network (sqlite3 database) in Torch7.
   ]],
   homepage = "http://github.com/deltheil/loadccv",
   license = "MIT/X11",
}

dependencies = {
   "torch >= 7.0",
   "nn",
   "lsqlite3",
   "penlight",
}

build = {
   type = "builtin",
   modules = {
      ['loadccv.init'] = 'init.lua',
      ['loadccv'] = 'util.c'
   },
   install = {
      bin = {
         loadccv = 'cli.lua',
      }
   },
}
