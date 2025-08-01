add_library(ActsPythonUtilities INTERFACE)
target_include_directories(ActsPythonUtilities INTERFACE include)

set(_python_dir "${CMAKE_BINARY_DIR}/python")
file(MAKE_DIRECTORY ${_python_dir})
file(MAKE_DIRECTORY ${_python_dir}/acts)

set(_python_install_dir "python/acts")

pybind11_add_module(ActsPythonBindings
  src/ModuleEntry.cpp
  src/Framework.cpp
  src/Base.cpp
  src/Alignment.cpp
  src/Detector.cpp
  src/Digitization.cpp
  src/Material.cpp
  src/Geometry.cpp
  src/GeometryBuildingGen1.cpp
  src/Blueprint.cpp
  src/ExampleAlgorithms.cpp
  src/MagneticField.cpp
  src/Output.cpp
  src/Input.cpp
  src/Propagation.cpp
  src/Navigation.cpp
  src/Generators.cpp
  src/Obj.cpp
  src/TruthTracking.cpp
  src/TrackFitting.cpp
  src/TrackFinding.cpp
  src/Vertexing.cpp
  src/AmbiguityResolution.cpp
  src/EventData.cpp
  src/Utilities.cpp
)
install(TARGETS ActsPythonBindings DESTINATION ${_python_install_dir})

set_target_properties(
    ActsPythonBindings
    PROPERTIES INSTALL_RPATH "\$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
)
set_target_properties(
    ActsPythonBindings
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_python_dir}/acts
)

target_link_libraries(
    ActsPythonBindings
    PUBLIC
        ActsCore
        ActsPythonUtilities
        ActsExamplesFramework
        ActsExamplesGeometry
        ActsExamplesDetectorGeneric
        ActsExamplesDetectorTGeo
        ActsExamplesMagneticField
        ActsExamplesIoCsv
        ActsExamplesIoObj
        ActsExamplesIoJson
        ActsExamplesGenerators
        ActsExamplesPrinters
        ActsExamplesTrackFinding
        ActsExamplesTrackFitting
        ActsExamplesVertexing
        ActsExamplesFatras
        ActsExamplesDetectorTelescope
        ActsExamplesUtilities
        ActsExamplesAmbiguityResolution
        ActsExamplesTruthTracking
        ActsExamplesDigitization
        ActsExamplesPropagation
        ActsExamplesMaterialMapping
)

set(py_files
    __init__.py
    examples/__init__.py
    examples/simulation.py
    examples/reconstruction.py
    examples/itk.py
    examples/odd.py
    _adapter.py
)

if(ACTS_BUILD_EXAMPLES_ROOT)
    target_link_libraries(ActsPythonBindings PUBLIC ActsExamplesIoRoot)
    target_sources(
        ActsPythonBindings
        PRIVATE src/RootInput.cpp src/RootOutput.cpp
    )
else()
    target_sources(ActsPythonBindings PRIVATE src/RootStub.cpp)
endif()

if(ACTS_BUILD_PLUGIN_JSON)
    target_link_libraries(ActsPythonBindings PUBLIC ActsPluginJson)
    target_sources(ActsPythonBindings PRIVATE src/Json.cpp)
else()
    target_sources(ActsPythonBindings PRIVATE src/JsonStub.cpp)
endif()

if(ACTS_BUILD_PLUGIN_GEOMODEL)
    target_link_libraries(
        ActsPythonBindings
        PUBLIC
            ActsPluginGeoModel
            ActsExamplesDetectorGeoModel
            ActsExamplesMuonSpectrometerMockupDetector
    )
    target_sources(ActsPythonBindings PRIVATE src/GeoModel.cpp)
else()
    target_sources(ActsPythonBindings PRIVATE src/GeoModelStub.cpp)
endif()

if(ACTS_BUILD_PLUGIN_ROOT)
    target_link_libraries(ActsPythonBindings PUBLIC ActsPluginRoot)
    target_sources(ActsPythonBindings PRIVATE src/TGeo.cpp)
else()
    target_sources(ActsPythonBindings PRIVATE src/TGeoStub.cpp)
endif()

if(ACTS_BUILD_PLUGIN_TRACCC)
    target_link_libraries(ActsPythonBindings PUBLIC ActsPluginDetray)
    target_sources(ActsPythonBindings PRIVATE src/Detray.cpp)
    target_link_libraries(ActsPythonBindings PUBLIC ActsPluginCovfie)
    target_sources(ActsPythonBindings PRIVATE src/Covfie.cpp)
    target_link_libraries(ActsPythonBindings PUBLIC ActsExamplesTraccc)
    target_sources(ActsPythonBindings PRIVATE src/Traccc.cpp)
else()
    target_sources(ActsPythonBindings PRIVATE src/DetrayStub.cpp)
    target_sources(ActsPythonBindings PRIVATE src/CovfieStub.cpp)
    target_sources(ActsPythonBindings PRIVATE src/TracccStub.cpp)
endif()

if(ACTS_BUILD_PLUGIN_ACTSVG)
    target_link_libraries(ActsPythonBindings PUBLIC ActsExamplesIoSvg)
    target_sources(ActsPythonBindings PRIVATE src/Svg.cpp)
else()
    target_sources(ActsPythonBindings PRIVATE src/SvgStub.cpp)
endif()

if(ACTS_BUILD_PLUGIN_DD4HEP AND ACTS_BUILD_EXAMPLES_DD4HEP)
    pybind11_add_module(ActsPythonBindingsDD4hep src/DD4hepComponent.cpp)
    target_link_libraries(
        ActsPythonBindingsDD4hep
        PUBLIC ActsPythonUtilities ActsExamplesDetectorDD4hep
    )
    add_dependencies(ActsPythonBindings ActsPythonBindingsDD4hep)

    install(TARGETS ActsPythonBindingsDD4hep DESTINATION ${_python_install_dir})
    set_target_properties(
        ActsPythonBindingsDD4hep
        PROPERTIES INSTALL_RPATH "\$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
    )
    set_target_properties(
        ActsPythonBindingsDD4hep
        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_python_dir}/acts
    )
    list(APPEND py_files examples/dd4hep.py)
endif()

if(ACTS_BUILD_EXAMPLES_PYTHIA8)
    target_link_libraries(
        ActsPythonBindings
        PUBLIC ActsExamplesGeneratorsPythia8
    )
    target_sources(ActsPythonBindings PRIVATE src/Pythia8.cpp)
else()
    target_sources(ActsPythonBindings PRIVATE src/Pythia8Stub.cpp)
endif()

if(ACTS_BUILD_EXAMPLES_GEANT4)
    pybind11_add_module(ActsPythonBindingsGeant4 src/Geant4Component.cpp)
    target_link_libraries(
        ActsPythonBindingsGeant4
        PUBLIC
            ActsExamplesGeant4
            ActsExamplesDetectorGeant4
            ActsPythonUtilities
            ActsExamplesMuonSpectrometerMockupDetector
    )
    add_dependencies(ActsPythonBindings ActsPythonBindingsGeant4)

    install(TARGETS ActsPythonBindingsGeant4 DESTINATION ${_python_install_dir})
    set_target_properties(
        ActsPythonBindingsGeant4
        PROPERTIES INSTALL_RPATH "\$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
    )
    set_target_properties(
        ActsPythonBindingsGeant4
        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_python_dir}/acts
    )
    list(APPEND py_files examples/geant4/__init__.py)
endif()

target_link_libraries(ActsPythonBindings PUBLIC ActsExamplesIoHepMC3)
target_sources(ActsPythonBindings PRIVATE src/HepMC3.cpp)
list(APPEND py_files examples/hepmc3.py)

if(ACTS_BUILD_EXAMPLES_EXATRKX)
    target_link_libraries(
        ActsPythonBindings
        PUBLIC ActsExamplesTrackFindingExaTrkX
    )
    target_sources(ActsPythonBindings PRIVATE src/ExaTrkXTrackFinding.cpp)
else()
    target_sources(ActsPythonBindings PRIVATE src/ExaTrkXTrackFindingStub.cpp)
endif()

if(ACTS_BUILD_EXAMPLES_EDM4HEP)
    pybind11_add_module(ActsPythonBindingsEDM4hep src/EDM4hepComponent.cpp)
    target_link_libraries(
        ActsPythonBindingsEDM4hep
        PUBLIC ActsPythonUtilities ActsExamplesIoEDM4hep
    )
    add_dependencies(ActsPythonBindings ActsPythonBindingsEDM4hep)
    install(
        TARGETS ActsPythonBindingsEDM4hep DESTINATION ${_python_install_dir}
    )
    set_target_properties(
        ActsPythonBindingsEDM4hep
        PROPERTIES INSTALL_RPATH "\$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
    )
    set_target_properties(
        ActsPythonBindingsEDM4hep
        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_python_dir}/acts
    )
    list(APPEND py_files examples/edm4hep.py)
endif()

if(ACTS_BUILD_EXAMPLES_PODIO)
    pybind11_add_module(ActsPythonBindingsPodio src/PodioComponent.cpp)
    target_link_libraries(
        ActsPythonBindingsPodio
        PUBLIC ActsPythonUtilities ActsExamplesIoPodio
    )
    add_dependencies(ActsPythonBindings ActsPythonBindingsPodio)
    install(TARGETS ActsPythonBindingsPodio DESTINATION ${_python_install_dir})
    set_target_properties(
        ActsPythonBindingsPodio
        PROPERTIES INSTALL_RPATH "\$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
    )
    set_target_properties(
        ActsPythonBindingsPodio
        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_python_dir}/acts
    )
    list(APPEND py_files examples/podio.py)
endif()

if(ACTS_BUILD_PLUGIN_ONNX)
    target_link_libraries(
        ActsPythonBindings
        PUBLIC ActsExamplesFrameworkML ActsExamplesTrackFindingML
    )
    target_sources(ActsPythonBindings PRIVATE src/Onnx.cpp)
    target_sources(ActsPythonBindings PRIVATE src/OnnxNeuralCalibrator.cpp)
    list(APPEND py_files examples/onnx/__init__.py)
else()
    target_sources(ActsPythonBindings PRIVATE src/OnnxStub.cpp)
    target_sources(ActsPythonBindings PRIVATE src/OnnxNeuralCalibratorStub.cpp)
endif()

if(ACTS_BUILD_PLUGIN_FASTJET)
    target_link_libraries(ActsPythonBindings PUBLIC ActsExamplesJets)
    target_sources(ActsPythonBindings PRIVATE src/TruthJet.cpp)
else()
    target_sources(ActsPythonBindings PRIVATE src/TruthJetStub.cpp)
endif()

if(ACTS_BUILD_EXAMPLES_HASHING)
    target_link_libraries(ActsPythonBindings PUBLIC ActsPluginHashing)
    target_sources(ActsPythonBindings PRIVATE src/Hashing.cpp)
    list(APPEND py_files examples/hashing.py)
else()
    target_sources(ActsPythonBindings PRIVATE src/HashingStub.cpp)
endif()

configure_file(setup.sh.in ${_python_dir}/setup.sh @ONLY)
install(FILES ${_python_dir}/setup.sh DESTINATION "python")

foreach(f ${py_files})
    set(_target ${_python_dir}/acts/${f})
    get_filename_component(_dir ${_target} DIRECTORY)
    file(MAKE_DIRECTORY ${_dir})

    file(
        CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/python/acts/${f} ${_target}
        SYMBOLIC
    )

    get_filename_component(_rel ${f} DIRECTORY)

    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/acts/${f}
        DESTINATION ${_python_install_dir}/${_rel}
    )
endforeach()
