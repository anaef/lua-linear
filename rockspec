rockspec_format = "3.0"
package = "lua-linear"
version = "1.0.0-1"
description = {
	summary = "Linear algebra for Lua",
	detailed = [[
		Lua Linear provides comprehensive linear algebra support for the Lua programming
		language. Where applicable, the BLAS and LAPACK implementations on the system are
		used.
	]],
	license = "MIT",
	homepage = "https://github.com/anaef/lua-linear",
	labels = { "math" },
}
dependencies = {
	"lua >= 5.1"
}
external_dependencies = {
	LIBBLAS = {
		library = "blas"
	},
	LIBLAPACKE = {
		header = "lapacke.h"
	},
}
source = {
	url = "git+https://github.com/anaef/lua-linear.git",
	tag = "v1.0.0",
}
build = {
	type = "builtin",
	modules = {
		linear = {
			sources = {
				"src/linear_core.c",
				"src/linear_elementary.c",
				"src/linear_unary.c",
				"src/linear_binary.c",
				"src/linear_program.c",
			},
			defines = {
				"_REENTRANT",
				"_GNU_SOURCE",
				"LINEAR_USE_AXPBY=1"
			},
			libraries = {
				"m",
				"blas",
				"lapacke",
			},
			incdirs = {
				"$(LIBBLAS_INCDIR)",
				"$(LIBLAPACKE_INCDIR)"
			},
		},
	},
}
