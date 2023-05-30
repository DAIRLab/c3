load("//tools/workspace/osqp:repository.bzl", "osqp_repository")  # noqa

def add_default_repositories(excludes = [], mirrors = DEFAULT_MIRRORS):
    """Declares workspace repositories for all externals needed by drake (other
    than those built into Bazel, of course).  This is intended to be loaded and
    called from a WORKSPACE file.

    Args:
        excludes: list of string names of repositories to exclude; this can
          be useful if a WORKSPACE file has already supplied its own external
          of a given name.
    """
    if "abseil_cpp_internal" not in excludes:
        abseil_cpp_internal_repository(name = "abseil_cpp_internal", mirrors = mirrors)  # noqa
    if "bazel_skylib" not in excludes:
        bazel_skylib_repository(name = "bazel_skylib", mirrors = mirrors)
    if "blas" not in excludes:
        blas_repository(name = "blas")
    if "buildifier" not in excludes:
        buildifier_repository(name = "buildifier", mirrors = mirrors)
    if "cc" not in excludes:
        cc_repository(name = "cc")
    if "ccd_internal" not in excludes:
        ccd_internal_repository(name = "ccd_internal", mirrors = mirrors)
    if "clang_cindex_python3_internal" not in excludes:
        clang_cindex_python3_internal_repository(name = "clang_cindex_python3_internal", mirrors = mirrors)  # noqa
    if "clp" not in excludes:
        clp_repository(name = "clp")
    if "com_jidesoft_jide_oss" not in excludes:
        com_jidesoft_jide_oss_repository(name = "com_jidesoft_jide_oss", mirrors = mirrors)  # noqa
    if "common_robotics_utilities" not in excludes:
        common_robotics_utilities_repository(name = "common_robotics_utilities", mirrors = mirrors)  # noqa
    if "commons_io" not in excludes:
        commons_io_repository(name = "commons_io", mirrors = mirrors)
    if "conex" not in excludes:
        conex_repository(name = "conex", mirrors = mirrors)
    if "csdp" not in excludes:
        csdp_repository(name = "csdp", mirrors = mirrors)
    if "curl_internal" not in excludes:
        curl_internal_repository(name = "curl_internal", mirrors = mirrors)
    if "double_conversion" not in excludes:
        double_conversion_repository(name = "double_conversion")
    if "doxygen" not in excludes:
        doxygen_repository(name = "doxygen", mirrors = mirrors)
    if "dm_control_internal" not in excludes:
        dm_control_internal_repository(name = "dm_control_internal", mirrors = mirrors)  # noqa
    if "drake_detected_os" not in excludes:
        os_repository(name = "drake_detected_os")
    if "drake_models" not in excludes:
        drake_models_repository(name = "drake_models", mirrors = mirrors)
    if "drake_visualizer" not in excludes:
        drake_visualizer_repository(name = "drake_visualizer", mirrors = mirrors)  # noqa
    if "eigen" not in excludes:
        eigen_repository(name = "eigen")
    if "expat" not in excludes:
        expat_repository(name = "expat")
    if "fcl_internal" not in excludes:
        fcl_internal_repository(name = "fcl_internal", mirrors = mirrors)
    if "fmt" not in excludes:
        fmt_repository(name = "fmt", mirrors = mirrors)
    if "gflags" not in excludes:
        gflags_repository(name = "gflags", mirrors = mirrors)
    if "gfortran" not in excludes:
        gfortran_repository(name = "gfortran")
    if "github3_py_internal" not in excludes:
        github3_py_internal_repository(name = "github3_py_internal", mirrors = mirrors)  # noqa
    if "glew" not in excludes:
        glew_repository(name = "glew")
    if "glib" not in excludes:
        glib_repository(name = "glib")
    if "glx" not in excludes:
        glx_repository(name = "glx")
    if "googlebenchmark" not in excludes:
        googlebenchmark_repository(name = "googlebenchmark", mirrors = mirrors)
    if "gtest" not in excludes:
        gtest_repository(name = "gtest", mirrors = mirrors)
    if "gurobi" not in excludes:
        gurobi_repository(name = "gurobi")
    if "gz_math_internal" not in excludes:
        gz_math_internal_repository(name = "gz_math_internal", mirrors = mirrors)  # noqa
    if "gz_utils_internal" not in excludes:
        gz_utils_internal_repository(name = "gz_utils_internal", mirrors = mirrors)  # noqa
    if "gym_py" not in excludes:
        gym_py_repository(name = "gym_py", mirrors = mirrors)
    if "intel_realsense_ros_internal" not in excludes:
        intel_realsense_ros_internal_repository(name = "intel_realsense_ros_internal", mirrors = mirrors)  # noqa
    if "ipopt" not in excludes:
        ipopt_repository(name = "ipopt")
    if "ipopt_internal_fromsource" not in excludes:
        ipopt_internal_fromsource_repository(name = "ipopt_internal_fromsource", mirrors = mirrors)  # noqa
    if "ipopt_internal_pkgconfig" not in excludes:
        ipopt_internal_pkgconfig_repository(name = "ipopt_internal_pkgconfig")
    if "lapack" not in excludes:
        lapack_repository(name = "lapack")
    if "lcm" not in excludes:
        lcm_repository(name = "lcm", mirrors = mirrors)
    if "libblas" not in excludes:
        libblas_repository(name = "libblas")
    if "libcmaes" not in excludes:
        libcmaes_repository(name = "libcmaes", mirrors = mirrors)
    if "libjpeg" not in excludes:
        libjpeg_repository(name = "libjpeg")
    if "liblapack" not in excludes:
        liblapack_repository(name = "liblapack")
    if "liblz4" not in excludes:
        liblz4_repository(name = "liblz4")
    if "liblzma" not in excludes:
        liblzma_repository(name = "liblzma")
    if "libpfm" not in excludes:
        libpfm_repository(name = "libpfm")
    if "libpng" not in excludes:
        libpng_repository(name = "libpng")
    if "libtiff" not in excludes:
        libtiff_repository(name = "libtiff")
    if "meshcat" not in excludes:
        meshcat_repository(name = "meshcat", mirrors = mirrors)
    if "mosek" not in excludes:
        mosek_repository(name = "mosek")
    if "msgpack_internal" not in excludes:
        msgpack_internal_repository(name = "msgpack_internal", mirrors = mirrors)  # noqa
    if "mumps_internal" not in excludes:
        mumps_internal_repository(name = "mumps_internal")
    if "mypy_extensions_internal" not in excludes:
        mypy_extensions_internal_repository(name = "mypy_extensions_internal", mirrors = mirrors)  # noqa
    if "mypy_internal" not in excludes:
        mypy_internal_repository(name = "mypy_internal", mirrors = mirrors)
    if "nanoflann_internal" not in excludes:
        nanoflann_internal_repository(name = "nanoflann_internal", mirrors = mirrors)  # noqa
    if "net_sf_jchart2d" not in excludes:
        net_sf_jchart2d_repository(name = "net_sf_jchart2d", mirrors = mirrors)
    if "nlopt_internal" not in excludes:
        nlopt_internal_repository(name = "nlopt_internal", mirrors = mirrors)
    if "openblas" not in excludes:
        openblas_repository(name = "openblas")
    if "opencl" not in excludes:
        opencl_repository(name = "opencl")
    if "opengl" not in excludes:
        opengl_repository(name = "opengl")
    if "optitrack_driver" not in excludes:
        optitrack_driver_repository(name = "optitrack_driver", mirrors = mirrors)  # noqa
    if "org_apache_xmlgraphics_commons" not in excludes:
        org_apache_xmlgraphics_commons_repository(name = "org_apache_xmlgraphics_commons", mirrors = mirrors)  # noqa
    if "osqp_internal" not in excludes:
        osqp_internal_repository(name = "osqp_internal", mirrors = mirrors)
    if "petsc" not in excludes:
        petsc_repository(name = "petsc", mirrors = mirrors)
    if "picosha2" not in excludes:
        picosha2_repository(name = "picosha2", mirrors = mirrors)
    if "platforms" not in excludes:
        platforms_repository(name = "platforms", mirrors = mirrors)
    if "pybind11" not in excludes:
        pybind11_repository(name = "pybind11", mirrors = mirrors)
    if "pycodestyle" not in excludes:
        pycodestyle_repository(name = "pycodestyle", mirrors = mirrors)
    if "python" not in excludes:
        python_repository(name = "python")
    if "qdldl_internal" not in excludes:
        qdldl_internal_repository(name = "qdldl_internal", mirrors = mirrors)
    if "qhull_internal" not in excludes:
        qhull_internal_repository(name = "qhull_internal", mirrors = mirrors)
    if "ros_xacro_internal" not in excludes:
        ros_xacro_internal_repository(name = "ros_xacro_internal", mirrors = mirrors)  # noqa
    if "rules_pkg" not in excludes:
        rules_pkg_repository(name = "rules_pkg", mirrors = mirrors)
    if "rules_python" not in excludes:
        rules_python_repository(name = "rules_python", mirrors = mirrors)
    if "scs_internal" not in excludes:
        scs_internal_repository(name = "scs_internal", mirrors = mirrors)
    if "sdformat_internal" not in excludes:
        sdformat_internal_repository(name = "sdformat_internal", mirrors = mirrors)  # noqa
    if "snopt" not in excludes:
        snopt_repository(name = "snopt")
    if "spdlog" not in excludes:
        spdlog_repository(name = "spdlog", mirrors = mirrors)
    if "stable_baselines3_internal" not in excludes:
        stable_baselines3_internal_repository(name = "stable_baselines3_internal", mirrors = mirrors)  # noqa
    if "statsjs" not in excludes:
        statsjs_repository(name = "statsjs", mirrors = mirrors)
    if "stduuid_internal" not in excludes:
        stduuid_internal_repository(name = "stduuid_internal", mirrors = mirrors)  # noqa
    if "styleguide" not in excludes:
        styleguide_repository(name = "styleguide", mirrors = mirrors)
    if "suitesparse_internal" not in excludes:
        suitesparse_internal_repository(name = "suitesparse_internal", mirrors = mirrors)  # noqa
    if "tinyobjloader" not in excludes:
        tinyobjloader_repository(name = "tinyobjloader", mirrors = mirrors)
    if "tinyxml2_internal" not in excludes:
        tinyxml2_internal_repository(name = "tinyxml2_internal", mirrors = mirrors)  # noqa
    if "tomli_internal" not in excludes:
        tomli_internal_repository(name = "tomli_internal", mirrors = mirrors)
    if "typing_extensions_internal" not in excludes:
        typing_extensions_internal_repository(name = "typing_extensions_internal", mirrors = mirrors)  # noqa
    if "uritemplate_py_internal" not in excludes:
        uritemplate_py_internal_repository(name = "uritemplate_py_internal", mirrors = mirrors)  # noqa
    if "usockets" not in excludes:
        usockets_repository(name = "usockets", mirrors = mirrors)
    if "uwebsockets" not in excludes:
        uwebsockets_repository(name = "uwebsockets", mirrors = mirrors)
    if "voxelized_geometry_tools" not in excludes:
        voxelized_geometry_tools_repository(name = "voxelized_geometry_tools", mirrors = mirrors)  # noqa
    if "vtk" not in excludes:
        vtk_repository(name = "vtk", mirrors = mirrors)
    if "x11" not in excludes:
        x11_repository(name = "x11")
    if "xmlrunner_py" not in excludes:
        xmlrunner_py_repository(name = "xmlrunner_py", mirrors = mirrors)
    if "yaml_cpp_internal" not in excludes:
        yaml_cpp_internal_repository(name = "yaml_cpp_internal", mirrors = mirrors)  # noqa
    if "zlib" not in excludes:
        zlib_repository(name = "zlib")

def add_default_toolchains(excludes = []):
    """Register toolchains for each language (e.g., "py") not explicitly
    excluded and/or not using an automatically generated toolchain.

    Args:
        excludes: List of languages for which a toolchain should not be
            registered.
    """

    if "py" not in excludes:
        # The Python debug toolchain on Linux is not loaded automatically, but
        # may be used by specifying the command line option
        # --extra_toolchains=//tools/py_toolchain:linux_dbg_toolchain
        native.register_toolchains(
            "@drake//tools/py_toolchain:linux_toolchain",
            "@drake//tools/py_toolchain:macos_i386_toolchain",
            "@drake//tools/py_toolchain:macos_arm64_toolchain",
        )

def add_default_workspace(
        repository_excludes = [],
        toolchain_excludes = [],
        mirrors = DEFAULT_MIRRORS):
    """Declare repositories in this WORKSPACE for each dependency of @drake
    (e.g., "eigen") that is not explicitly excluded, and register toolchains
    for each language (e.g., "py") not explicitly excluded and/or not using an
    automatically generated toolchain.

    Args:
        repository_excludes: List of repositories that should not be declared
            in this WORKSPACE.
        toolchain_excludes: List of languages for which a toolchain should not
            be registered.
        mirrors: Dictionary of mirrors from which to download repository files.
            See mirrors.bzl file in this directory for the file format and
            default values.
    """

    add_default_repositories(excludes = repository_excludes, mirrors = mirrors)
    add_default_toolchains(excludes = toolchain_excludes)
