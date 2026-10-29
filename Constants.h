/*
* This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CONSTANTSH
#define CONSTANTSH

// Release number
#define RELEASE "2.2-opt.1"

// Use symmetry
//#define USE_SYMMETRY

// Number of random jumps
// Max 512 for the GPU
#define NB_JUMP 32

// GPU group size (overridable at compile time for A/B benchmarking)
#ifndef GPU_GRP_SIZE
#define GPU_GRP_SIZE 128
#endif

// Default maximum GPU steps per kernel call (overridable at compile time;
// the runtime may choose a smaller batch to protect the DP output buffer).
#ifndef NB_RUN
#define NB_RUN 64
#endif

// Kangaroo type
#define TAME 0  // Tame kangaroo
#define WILD 1  // Wild kangaroo

// SendDP Period in sec
#define SEND_PERIOD 2.0

// Timeout before closing connection idle client in sec
#define CLIENT_TIMEOUT 3600.0

// Number of merge partition
#define MERGE_PART 256

#endif //CONSTANTSH
