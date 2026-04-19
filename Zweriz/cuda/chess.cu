// Zweriz Chess Engine - cuda/chess.cu
// by toiabzahoor

#define MAX_QUEUE_SIZE 16384 
#define MAX_MOVES 218

__device__ bool is_in_check(const float* board, float my_sign) {
    unsigned long long K = 0, opp_K = 0, opp_P = 0, opp_N = 0, opp_RQ = 0, opp_BQ = 0, empty = 0;
    float enemy_sign = -my_sign;
    
    for (int i = 0; i < 64; i++) {
        float p = board[i];
        unsigned long long mask = (1ULL << i);
        if (p == 0.0f) empty |= mask;
        else if (p == 6.0f * my_sign) K |= mask;
        else if (p == 6.0f * enemy_sign) opp_K |= mask;
        else if (p == 1.0f * enemy_sign) opp_P |= mask;
        else if (p == 2.0f * enemy_sign) opp_N |= mask;
        else if (p == 4.0f * enemy_sign || p == 5.0f * enemy_sign) opp_RQ |= mask;
        else if (p == 3.0f * enemy_sign || p == 5.0f * enemy_sign) opp_BQ |= mask;
    }

    if (K == 0) return true;

    if (((K >> 1) & 0x7F7F7F7F7F7F7F7FULL) & opp_K) return true;
    if (((K << 1) & 0xFEFEFEFEFEFEFEFEULL) & opp_K) return true;
    if ((K >> 8) & opp_K) return true;
    if ((K << 8) & opp_K) return true;
    if (((K >> 9) & 0x7F7F7F7F7F7F7F7FULL) & opp_K) return true;
    if (((K >> 7) & 0xFEFEFEFEFEFEFEFEULL) & opp_K) return true;
    if (((K << 7) & 0x7F7F7F7F7F7F7F7FULL) & opp_K) return true;
    if (((K << 9) & 0xFEFEFEFEFEFEFEFEULL) & opp_K) return true;

    if (my_sign > 0.0f) { 
        if ((K >> 7) & opp_P & 0xFEFEFEFEFEFEFEFEULL) return true;
        if ((K >> 9) & opp_P & 0x7F7F7F7F7F7F7F7FULL) return true; 
    } else { 
        if ((K << 7) & opp_P & 0x7F7F7F7F7F7F7F7FULL) return true; 
        if ((K << 9) & opp_P & 0xFEFEFEFEFEFEFEFEULL) return true; 
    }

    if (((K >> 17) | (K << 15)) & opp_N & 0x7F7F7F7F7F7F7F7FULL) return true; 
    if (((K >> 15) | (K << 17)) & opp_N & 0xFEFEFEFEFEFEFEFEULL) return true;
    if (((K >> 10) | (K <<  6)) & opp_N & 0x3F3F3F3F3F3F3F3FULL) return true;
    if (((K >>  6) | (K << 10)) & opp_N & 0xFCFCFCFCFCFCFCFCULL) return true;

    unsigned long long ray_N = K, ray_S = K, ray_E = K, ray_W = K;
    unsigned long long ray_NE = K, ray_NW = K, ray_SE = K, ray_SW = K;

    for (int dist = 1; dist < 8; dist++) {
        ray_N = (ray_N >> 8); 
        ray_S = (ray_S << 8);
        ray_E = (ray_E << 1) & 0xFEFEFEFEFEFEFEFEULL;
        ray_W = (ray_W >> 1) & 0x7F7F7F7F7F7F7F7FULL;
        
        if (ray_N & opp_RQ) return true;
        if (ray_S & opp_RQ) return true;
        if (ray_E & opp_RQ) return true;
        if (ray_W & opp_RQ) return true;

        ray_N &= empty; ray_S &= empty; ray_E &= empty; ray_W &= empty;

        ray_NE = (ray_NE >> 7) & 0xFEFEFEFEFEFEFEFEULL;
        ray_NW = (ray_NW >> 9) & 0x7F7F7F7F7F7F7F7FULL;
        ray_SE = (ray_SE << 9) & 0xFEFEFEFEFEFEFEFEULL;
        ray_SW = (ray_SW << 7) & 0x7F7F7F7F7F7F7F7FULL;

        if (ray_NE & opp_BQ) return true;
        if (ray_NW & opp_BQ) return true;
        if (ray_SE & opp_BQ) return true;
        if (ray_SW & opp_BQ) return true;

        ray_NE &= empty; ray_NW &= empty; ray_SE &= empty; ray_SW &= empty;
    }
    return false;
}

extern "C" __global__ void init_search_kernel(
    const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size 
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float* initial_board = (const float*)in[0];
    float current_turn = ((const float*)in[1])[0];

    float* queue_boards = (float*)out[0];
    float* queue_state = (float*)out[1]; 
    float* queue_root_ids = (float*)out[2];
    float* queue_depths = (float*)out[3];
    float* queue_coords = (float*)out[4];
    float* queue_grace_ttl = (float*)out[5];
    float* root_stats = (float*)out[6];

    float my_sign = (current_turn == 0.0f) ? 1.0f : -1.0f;
    int pawn_dir = (current_turn == 0.0f) ? -1 : 1; 
    int move_count = 0;

    #define ADD_ROOT_MOVE(tr, tc) \
        if (move_count < MAX_QUEUE_SIZE) { \
            size_t offset = move_count * 64; \
            for (int j = 0; j < 64; j++) queue_boards[offset + j] = initial_board[j]; \
            queue_boards[offset + sq] = 0.0f; \
            queue_boards[offset + (tr * 8 + tc)] = p; \
            if (!is_in_check(&queue_boards[offset], my_sign)) { \
                queue_state[move_count] = 1.0f; \
                queue_root_ids[move_count] = (float)move_count; \
                queue_depths[move_count] = 1.0f; \
                queue_grace_ttl[move_count] = 0.0f; \
                queue_coords[move_count * 4 + 0] = (float)r; \
                queue_coords[move_count * 4 + 1] = (float)c; \
                queue_coords[move_count * 4 + 2] = (float)tr; \
                queue_coords[move_count * 4 + 3] = (float)tc; \
                root_stats[move_count * 5 + 0] = -1e38f; root_stats[move_count * 5 + 1] = 1e38f; \
                root_stats[move_count * 5 + 2] = 0.0f;   root_stats[move_count * 5 + 3] = 0.0f; \
                root_stats[move_count * 5 + 4] = 0.0f; \
                move_count++; \
            } \
        }

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int sq = r * 8 + c; float p = initial_board[sq];
            if (p == 0.0f || (p * my_sign) < 0.0f) continue;
            float abs_p = (p < 0.0f) ? -p : p;

            if (abs_p == 1.0f) {
                int tr = r + pawn_dir;
                if (tr >= 0 && tr < 8) {
                    if (initial_board[tr * 8 + c] == 0.0f) {
                        int tc = c; ADD_ROOT_MOVE(tr, tc);
                        bool is_start_rank = (current_turn == 0.0f && r == 6) || (current_turn == 1.0f && r == 1);
                        if (is_start_rank && initial_board[(r + pawn_dir*2) * 8 + c] == 0.0f) { int tc2 = c; ADD_ROOT_MOVE(r + pawn_dir*2, tc2); }
                    }
                    if (c > 0 && initial_board[tr * 8 + (c - 1)] * my_sign < 0.0f) { int tc = c - 1; ADD_ROOT_MOVE(tr, tc); }
                    if (c < 7 && initial_board[tr * 8 + (c + 1)] * my_sign < 0.0f) { int tc = c + 1; ADD_ROOT_MOVE(tr, tc); }
                }
            }
            else if (abs_p == 2.0f) {
                int dr[] = {-2, -2, -1, -1, 1, 1, 2, 2}; int dc[] = {-1, 1, -2, 2, -2, 2, -1, 1};
                for (int i = 0; i < 8; i++) {
                    int tr = r + dr[i]; int tc = c + dc[i];
                    if (tr >= 0 && tr < 8 && tc >= 0 && tc < 8) {
                        float dest = initial_board[tr * 8 + tc];
                        if (dest == 0.0f || (dest * my_sign) < 0.0f) { ADD_ROOT_MOVE(tr, tc); }
                    }
                }
            }
            else if (abs_p == 6.0f) {
                int dr[] = {-1, -1, -1, 0, 0, 1, 1, 1}; int dc[] = {-1, 0, 1, -1, 1, -1, 0, 1};
                for (int i = 0; i < 8; i++) {
                    int tr = r + dr[i]; int tc = c + dc[i];
                    if (tr >= 0 && tr < 8 && tc >= 0 && tc < 8) {
                        float dest = initial_board[tr * 8 + tc];
                        if (dest == 0.0f || (dest * my_sign) < 0.0f) { ADD_ROOT_MOVE(tr, tc); }
                    }
                }
            }
            else if (abs_p == 3.0f || abs_p == 4.0f || abs_p == 5.0f) {
                bool is_straight = (abs_p == 4.0f || abs_p == 5.0f); bool is_diagonal = (abs_p == 3.0f || abs_p == 5.0f);
                int dr[] = {-1, 1, 0, 0, -1, -1, 1, 1}; int dc[] = {0, 0, -1, 1, -1, 1, -1, 1};
                int start_idx = is_straight ? 0 : 4; int end_idx = is_diagonal ? 8 : 4;
                for (int i = start_idx; i < end_idx; i++) {
                    for (int step = 1; step < 8; step++) {
                        int tr = r + (dr[i] * step); int tc = c + (dc[i] * step);
                        if (tr < 0 || tr >= 8 || tc < 0 || tc >= 8) break; 
                        float dest = initial_board[tr * 8 + tc];
                        if (dest == 0.0f) { ADD_ROOT_MOVE(tr, tc); } 
                        else if ((dest * my_sign) < 0.0f) { ADD_ROOT_MOVE(tr, tc); break; } 
                        else break;
                    }
                }
            }
        }
    }
    for (int i = move_count; i < MAX_QUEUE_SIZE; i++) queue_state[i] = 0.0f;
}

extern "C" __global__ void expand_kernel(
    const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size 
) {
    __shared__ int s_move_count;
    if (threadIdx.x == 0) s_move_count = 0;

    const float* queue_boards = (const float*)in[0];
    float* queue_state = (float*)in[1];
    const float* queue_depths = (const float*)in[2];
    float current_turn = ((const float*)in[3])[0];
    float current_depth = ((const float*)in[4])[0];

    float* exp_boards = (float*)out[0];
    float* exp_active = (float*)out[1];

    int parent_idx = blockIdx.x; 
    if (queue_state[parent_idx] != 1.0f || queue_depths[parent_idx] != current_depth) return;

    for (int i = threadIdx.x; i < MAX_MOVES; i += blockDim.x) {
        exp_active[parent_idx * MAX_MOVES + i] = 0.0f;
    }
    __syncthreads();

    int sq = threadIdx.x; 
    float my_sign = (current_turn == 0.0f) ? 1.0f : -1.0f;
    int pawn_dir = (current_turn == 0.0f) ? -1 : 1; 

    if (sq < 64) {
        int r = sq / 8; int c = sq % 8;
        float p = queue_boards[parent_idx * 64 + sq];
        
        if (p != 0.0f && (p * my_sign) > 0.0f) {
            float abs_p = (p < 0.0f) ? -p : p;

            #define ADD_EXP_MOVE(tr, tc) \
                { \
                    int child_idx = atomicAdd(&s_move_count, 1); \
                    if (child_idx < MAX_MOVES) { \
                        size_t child_offset = (parent_idx * MAX_MOVES + child_idx) * 64; \
                        for (int j = 0; j < 64; j++) exp_boards[child_offset + j] = queue_boards[parent_idx * 64 + j]; \
                        exp_boards[child_offset + sq] = 0.0f; \
                        exp_boards[child_offset + ((tr) * 8 + (tc))] = p; \
                        if (!is_in_check(&exp_boards[child_offset], my_sign)) { \
                            exp_active[parent_idx * MAX_MOVES + child_idx] = 1.0f; \
                        } \
                    } \
                }

            if (abs_p == 1.0f) {
                int tr = r + pawn_dir;
                if (tr >= 0 && tr < 8) {
                    if (queue_boards[parent_idx * 64 + tr * 8 + c] == 0.0f) {
                        ADD_EXP_MOVE(tr, c);
                        bool is_start = (current_turn == 0.0f && r == 6) || (current_turn == 1.0f && r == 1);
                        if (is_start && queue_boards[parent_idx * 64 + (r + pawn_dir*2) * 8 + c] == 0.0f) { ADD_EXP_MOVE(r + pawn_dir*2, c); }
                    }
                    if (c > 0 && queue_boards[parent_idx * 64 + tr * 8 + (c - 1)] * my_sign < 0.0f) { ADD_EXP_MOVE(tr, c - 1); }
                    if (c < 7 && queue_boards[parent_idx * 64 + tr * 8 + (c + 1)] * my_sign < 0.0f) { ADD_EXP_MOVE(tr, c + 1); }
                }
            }
            else if (abs_p == 2.0f) {
                int dr[] = {-2, -2, -1, -1, 1, 1, 2, 2}; int dc[] = {-1, 1, -2, 2, -2, 2, -1, 1};
                for (int i = 0; i < 8; i++) {
                    int tr = r + dr[i]; int tc = c + dc[i];
                    if (tr >= 0 && tr < 8 && tc >= 0 && tc < 8) {
                        float dest = queue_boards[parent_idx * 64 + tr * 8 + tc];
                        if (dest == 0.0f || (dest * my_sign) < 0.0f) { ADD_EXP_MOVE(tr, tc); }
                    }
                }
            }
            else if (abs_p == 6.0f) {
                int dr[] = {-1, -1, -1, 0, 0, 1, 1, 1}; int dc[] = {-1, 0, 1, -1, 1, -1, 0, 1};
                for (int i = 0; i < 8; i++) {
                    int tr = r + dr[i]; int tc = c + dc[i];
                    if (tr >= 0 && tr < 8 && tc >= 0 && tc < 8) {
                        float dest = queue_boards[parent_idx * 64 + tr * 8 + tc];
                        if (dest == 0.0f || (dest * my_sign) < 0.0f) { ADD_EXP_MOVE(tr, tc); }
                    }
                }
            }
            else if (abs_p == 3.0f || abs_p == 4.0f || abs_p == 5.0f) {
                bool is_straight = (abs_p == 4.0f || abs_p == 5.0f); bool is_diagonal = (abs_p == 3.0f || abs_p == 5.0f);
                int dr[] = {-1, 1, 0, 0, -1, -1, 1, 1}; int dc[] = {0, 0, -1, 1, -1, 1, -1, 1};
                int start_idx = is_straight ? 0 : 4; int end_idx = is_diagonal ? 8 : 4;
                for (int i = start_idx; i < end_idx; i++) {
                    for (int step = 1; step < 8; step++) {
                        int tr = r + (dr[i] * step); int tc = c + (dc[i] * step);
                        if (tr < 0 || tr >= 8 || tc < 0 || tc >= 8) break; 
                        float dest = queue_boards[parent_idx * 64 + tr * 8 + tc];
                        if (dest == 0.0f) { ADD_EXP_MOVE(tr, tc); } 
                        else if ((dest * my_sign) < 0.0f) { ADD_EXP_MOVE(tr, tc); break; } 
                        else break;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) queue_state[parent_idx] = 2.0f; 
}

extern "C" __global__ void commit_kernel(
    const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size 
) {
    const float* exp_boards = (const float*)in[0];
    const float* exp_scores = (const float*)in[1];
    const float* exp_active = (const float*)in[2];
    const float* queue_depths = (const float*)in[3];
    const float* queue_grace_ttl = (const float*)in[4];
    const float* queue_root_ids = (const float*)in[5];
    const float* thresholds = (const float*)in[6];
    const float* queue_state_in = (const float*)in[7];
    float root_turn = ((const float*)in[8])[0];
    float sim_turn = ((const float*)in[9])[0];
    float north_star = ((const float*)in[10])[0]; 

    float* queue_boards = (float*)out[0];
    float* queue_scores = (float*)out[1];
    float* queue_depths_out = (float*)out[2];
    float* queue_state_out = (float*)out[3];
    float* queue_grace_ttl_out = (float*)out[4];
    float* queue_root_ids_out = (float*)out[5];
    float* root_stats = (float*)out[6];

    int parent_idx = blockIdx.x;
    int tid = threadIdx.x; 

    if (queue_state_in[parent_idx] != 2.0f) return; 

    if (tid < MAX_MOVES && exp_active[parent_idx * MAX_MOVES + tid] == 1.0f) {
        float my_val = exp_scores[parent_idx * MAX_MOVES + tid];
        
        float delta = north_star - my_val;
        bool keep = false; 
        float new_ttl = 0.0f;

        if (queue_grace_ttl[parent_idx] > 0.0f) {
            new_ttl = queue_grace_ttl[parent_idx] - 1.0f;
            if (new_ttl > 0.0f) keep = true;
        } else {
            if (delta <= thresholds[0]) { keep = true; new_ttl = 0.0f; } 
            else if (delta >= thresholds[1]) { keep = true; new_ttl = thresholds[2]; }
        }

        if (keep) {
            int dest_idx = -1;
            int start_scan = (parent_idx * 17 + tid * 7) % MAX_QUEUE_SIZE; 
            for (int k = 0; k < MAX_QUEUE_SIZE; k++) {
                int i = (start_scan + k) % MAX_QUEUE_SIZE;
                int* state_ptr = (int*)&queue_state_out[i];
                if (atomicCAS(state_ptr, 0, __float_as_int(1.0f)) == 0) { dest_idx = i; break; }
            }
            
            if (dest_idx != -1) {
                size_t child_offset = (parent_idx * MAX_MOVES + tid) * 64;
                for (int j = 0; j < 64; j++) queue_boards[dest_idx * 64 + j] = exp_boards[child_offset + j];
                queue_scores[dest_idx] = my_val;
                queue_depths_out[dest_idx] = queue_depths[parent_idx] + 1.0f;
                queue_grace_ttl_out[dest_idx] = new_ttl;
                queue_root_ids_out[dest_idx] = queue_root_ids[parent_idx];
            }

            float root_multiplier = (sim_turn == root_turn) ? 1.0f : -1.0f;
            float root_val = my_val * root_multiplier;

            int root_id = (int)queue_root_ids[parent_idx];
            atomicMaxFloat(&root_stats[root_id * 5 + 0], root_val);
            atomicMinFloat(&root_stats[root_id * 5 + 1], root_val);
            atomicAddFloat(&root_stats[root_id * 5 + 2], root_val);
            atomicAddFloat(&root_stats[root_id * 5 + 3], 1.0f);
            atomicAddFloat(&root_stats[root_id * 5 + 4], delta); 
        }
    }

    __syncthreads();
    if (tid == 0) queue_state_out[parent_idx] = 0.0f; 
}
