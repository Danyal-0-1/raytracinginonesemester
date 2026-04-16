cmake_minimum_required(VERSION 3.16)
project(renderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# nlohmann/json (scene config)
# -----------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json)

# -----------------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------------
add_executable(renderer
        src/render.cpp
        src/MeshOBJ.cpp
        src/scene_loader.cpp
)
target_include_directories(renderer PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(renderer PRIVATE nlohmann_json::nlohmann_json)

find_package(glm QUIET)
if(glm_FOUND)
  target_link_libraries(renderer PRIVATE glm::glm)
endif()

# -----------------------------------------------------------------------------
# StagePreview: camera/pixel/ray/mesh visualization (Polyscope)
# -----------------------------------------------------------------------------
FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG        v2.5.0
)
FetchContent_MakeAvailable(polyscope)

add_executable(StagePreview
        src/stage_preview.cpp
        src/MeshOBJ.cpp
        src/scene_loader.cpp
)

target_include_directories(StagePreview PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(StagePreview PRIVATE polyscope nlohmann_json::nlohmann_json)
if(glm_FOUND)
  target_link_libraries(StagePreview PRIVATE glm::glm)
endif()