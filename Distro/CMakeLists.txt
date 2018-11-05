cmake_minimum_required(VERSION 3.12)
project(Distro C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(Distro
        ParticleFilters.c
        ParticleFilters.h
        ParticleUtils.h)
