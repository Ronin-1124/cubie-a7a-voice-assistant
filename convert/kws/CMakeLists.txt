cmake_minimum_required(VERSION 2.8)
project(kws_npu_demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SYSTEM_PROCESSOR arm)

if(EXTERN_DEFINE_TARGET)
    message(STATUS "USE EXTERN DEFINE TARGET")
endif()
if(USE_EXTERN_TOOLCHAIN)
    message(STATUS "USE EXTERN TOOLCHAIN BUILD")
endif()

# examples/voice_assistant_a7a/convert_kws_npu/kws_demo -> zoo root
set(MODEL_ZOO_HOME_DIR ${CMAKE_SOURCE_DIR}/../../../..)
set(ZIPFORMER_DIR ${MODEL_ZOO_HOME_DIR}/examples/zipformer)

message(STATUS "CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
message(STATUS "TARGET_NAME = ${TARGET_NAME}")
message(STATUS "MODEL_ZOO_HOME_DIR = ${MODEL_ZOO_HOME_DIR}")

if(CMAKE_C_COMPILER MATCHES "android-ndk")
    set(SYS_ARCH android)
elseif(CMAKE_C_COMPILER MATCHES "aarch64")
    set(SYS_ARCH linux_aarch64)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
else()
    set(SYS_ARCH linux_armhf)
    add_definitions(-DLIBVIP_VERSION_85X)
endif()

set(LIBSNDFILE_DIR ${MODEL_ZOO_HOME_DIR}/3rdparty/libsndfile)
include_directories(${LIBSNDFILE_DIR}/include)
if(SYS_ARCH STREQUAL "linux_aarch64")
    set(LIBSNDFILE_LIB ${LIBSNDFILE_DIR}/Linux/aarch64/libsndfile.a)
else()
    message(FATAL_ERROR "This demo is built for linux_aarch64")
endif()

set(NPU_RT_DIR ${MODEL_ZOO_HOME_DIR}/common/npuruntime)
file(GLOB NPU_RT_LIBS "${NPU_RT_DIR}/lib_${SYS_ARCH}/${TARGET_NAME}/*.so*")

aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC)
aux_source_directory(${NPU_RT_DIR} NPULIB_DIRSRCS)
aux_source_directory(${ZIPFORMER_DIR}/kaldi-native-fbank/csrc/src KALDI_NATIVE_FBANK_SRC)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ZIPFORMER_DIR}/kaldi-native-fbank/csrc/include)
include_directories(./ ${NPU_RT_DIR} ${NPU_RT_DIR}/include)

string(TOLOWER "${TARGET_NAME}" TARGET_LOWER)
string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_LOWER)

add_executable(${PROJECT_NAME}_${TARGET_LOWER} ${NPULIB_DIRSRCS} ${SRC} ${KALDI_NATIVE_FBANK_SRC})
target_link_libraries(${PROJECT_NAME}_${TARGET_LOWER} ${NPU_RT_LIBS} ${LIBSNDFILE_LIB} -lm -ldl)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${PROJECT_NAME}_${SYSTEM_LOWER}_${TARGET_LOWER})
install(TARGETS ${PROJECT_NAME}_${TARGET_LOWER} DESTINATION ./)
file(GLOB MODEL_FILE "model/*_${TARGET_LOWER}.nb")
file(GLOB TEST_WAV_FILES "model/*.wav")
file(GLOB VOCAB_FILES "model/tokens.txt" "model/keywords*.txt")
install(FILES ${MODEL_FILE} DESTINATION ./model/)
install(FILES ${TEST_WAV_FILES} DESTINATION ./model/)
install(FILES ${VOCAB_FILES} DESTINATION ./model/)
install(FILES ${NPU_RT_LIBS} DESTINATION ./lib/)
