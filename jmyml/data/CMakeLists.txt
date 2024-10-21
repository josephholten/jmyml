add_executable(IdxViewer IdxViewer.cpp)
target_link_libraries(IdxViewer PRIVATE flags)
target_link_libraries(IdxViewer PRIVATE raylib)
target_link_libraries(IdxViewer PRIVATE fmt::fmt)