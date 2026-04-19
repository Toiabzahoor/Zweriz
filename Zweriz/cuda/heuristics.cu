// Zweriz Chess Engine - cuda/heuristics.cu
// by toiabzahoor

__device__ inline void atomicAddFloat(float* addr, float val) { atomicAdd(addr, val); }

__device__ inline void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ inline void atomicMinFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__constant__ float PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

__constant__ float PST_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

__constant__ float PST_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};

__constant__ float PST_ROOK[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
};

__constant__ float PST_QUEEN[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};

__constant__ float PST_KING_MIDGAME[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};

__constant__ float PST_PAWN_ENDGAME[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
     80, 80, 80, 80, 80, 80, 80, 80,
     50, 50, 50, 50, 50, 50, 50, 50,
     30, 30, 30, 30, 30, 30, 30, 30,
     20, 20, 20, 20, 20, 20, 20, 20,
     10, 10, 10, 10, 10, 10, 10, 10,
      0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0
};

__constant__ float PST_KING_ENDGAME[64] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
};

#define BISHOP_PAIR_BONUS 30.0f
#define ROOK_SEMI_OPEN_FILE_BONUS 15.0f
#define ROOK_OPEN_FILE_BONUS 25.0f
#define DOUBLED_PAWN_PENALTY -15.0f
#define ISOLATED_PAWN_PENALTY -15.0f
#define PASSED_PAWN_BONUS_MG 20.0f
#define PASSED_PAWN_BONUS_EG 50.0f
#define ROOK_ON_7TH_BONUS 25.0f
#define KNIGHT_OUTPOST_BONUS 15.0f
#define KING_SHIELD_PENALTY -20.0f

#define GET_PAWNS(files, c) (((files) >> ((c) * 4)) & 0xF)
#define ADD_PAWN(files, c) ((files) += (1 << ((c) * 4)))

__device__ inline int center_manhattan_distance(int sq) {
    int r = sq / 8; int c = sq % 8;
    int r_dist = (r < 4) ? (3 - r) : (r - 4);
    int c_dist = (c < 4) ? (3 - c) : (c - 4);
    return r_dist + c_dist;
}

__device__ inline float mop_up(int winning_k, int losing_k, float winning_sign) {
    float eval = 0.0f;
    eval += (float)center_manhattan_distance(losing_k) * 10.0f;

    int w_r = winning_k / 8; int w_c = winning_k % 8;
    int l_r = losing_k / 8; int l_c = losing_k % 8;
    int dist = abs(w_r - l_r) + abs(w_c - l_c);
    eval -= (float)dist * 4.0f;
    return eval * winning_sign;
}

extern "C" __global__ void eval_kernel(
    const unsigned long long* in,
    unsigned long long* out,
    const unsigned long long* shapes,
    size_t size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float* child_boards = (const float*)in[0];
    const float* child_active = (const float*)in[1];
    float current_turn = ((const float*)in[2])[0];

    float* scores = (float*)out[0];
    float* global_best = (float*)out[1];

    if (child_active[idx] == 0.0f) {
        scores[idx] = -1e38f;
        return;
    }

    float mg_score = 0.0f;
    float eg_score = 0.0f;
    int phase = 0;

    float w_mat = 0.0f, b_mat = 0.0f;
    int w_k_sq = -1, b_k_sq = -1;

    unsigned int w_pawn_files = 0;
    unsigned int b_pawn_files = 0;
    int w_bishops = 0, b_bishops = 0;

    for (int i = 0; i < 64; i++) {
        float p = child_boards[idx * 64 + i];
        if (p == 0.0f) continue;

        float abs_p = fabsf(p);
        float sign = (p > 0.0f) ? 1.0f : -1.0f;
        int table_idx = (p > 0.0f) ? i : (63 - i);

        float mg_val = 0.0f; float eg_val = 0.0f;
        int col = i % 8;

        if (abs_p == 1.0f) {
            mg_val = 100.0f + PST_PAWN[table_idx];
            eg_val = 100.0f + PST_PAWN_ENDGAME[table_idx];
            if (p > 0.0f) { w_mat += 100.0f; ADD_PAWN(w_pawn_files, col); }
            else { b_mat += 100.0f; ADD_PAWN(b_pawn_files, col); }
        }
        else if (abs_p == 2.0f) {
            mg_val = 320.0f + PST_KNIGHT[table_idx]; eg_val = mg_val; phase += 1;
            if (p > 0.0f) w_mat += 320.0f; else b_mat += 320.0f;
        }
        else if (abs_p == 3.0f) {
            mg_val = 330.0f + PST_BISHOP[table_idx]; eg_val = mg_val; phase += 1;
            if (p > 0.0f) { w_mat += 330.0f; w_bishops++; }
            else { b_mat += 330.0f; b_bishops++; }
        }
        else if (abs_p == 4.0f) {
            mg_val = 500.0f + PST_ROOK[table_idx]; eg_val = mg_val; phase += 2;
            if (p > 0.0f) w_mat += 500.0f; else b_mat += 500.0f;
        }
        else if (abs_p == 5.0f) {
            mg_val = 900.0f + PST_QUEEN[table_idx]; eg_val = mg_val; phase += 4;
            if (p > 0.0f) w_mat += 900.0f; else b_mat += 900.0f;
        }
        else if (abs_p == 6.0f) {
            mg_val = 20000.0f + PST_KING_MIDGAME[table_idx];
            eg_val = 20000.0f + PST_KING_ENDGAME[table_idx];
            if (p > 0.0f) w_k_sq = i; else b_k_sq = i;
        }

        mg_score += mg_val * sign;
        eg_score += eg_val * sign;
    }

    for (int i = 0; i < 64; i++) {
        float p = child_boards[idx * 64 + i];
        if (p == 0.0f) continue;

        float abs_p = fabsf(p);
        int r = i / 8; int c = i % 8;
        bool is_white = (p > 0.0f);

        if (abs_p == 4.0f) {
            int w_p = GET_PAWNS(w_pawn_files, c);
            int b_p = GET_PAWNS(b_pawn_files, c);
            if (is_white) {
                if (w_p == 0 && b_p == 0) mg_score += ROOK_OPEN_FILE_BONUS;
                else if (w_p == 0) mg_score += ROOK_SEMI_OPEN_FILE_BONUS;
                if (r == 1) { mg_score += ROOK_ON_7TH_BONUS; eg_score += ROOK_ON_7TH_BONUS; }
            } else {
                if (w_p == 0 && b_p == 0) mg_score -= ROOK_OPEN_FILE_BONUS;
                else if (b_p == 0) mg_score -= ROOK_SEMI_OPEN_FILE_BONUS;
                if (r == 6) { mg_score -= ROOK_ON_7TH_BONUS; eg_score -= ROOK_ON_7TH_BONUS; }
            }
        }
        else if (abs_p == 2.0f) {
            int back_r = is_white ? r + 1 : r - 1;
            bool supported = false;
            if (back_r >= 0 && back_r < 8) {
                if (c > 0 && child_boards[idx * 64 + (back_r * 8 + c - 1)] == (is_white ? 1.0f : -1.0f)) supported = true;
                if (c < 7 && child_boards[idx * 64 + (back_r * 8 + c + 1)] == (is_white ? 1.0f : -1.0f)) supported = true;
            }
            if (supported) {
                if (is_white) { mg_score += KNIGHT_OUTPOST_BONUS; eg_score += KNIGHT_OUTPOST_BONUS; }
                else { mg_score -= KNIGHT_OUTPOST_BONUS; eg_score -= KNIGHT_OUTPOST_BONUS; }
            }
        }
        else if (abs_p == 1.0f) {
            int my_pawns = is_white ? w_pawn_files : b_pawn_files;
            int opp_pawns = is_white ? b_pawn_files : w_pawn_files;

            if (GET_PAWNS(my_pawns, c) > 1) {
                if (is_white) { mg_score += DOUBLED_PAWN_PENALTY; eg_score += DOUBLED_PAWN_PENALTY; }
                else { mg_score -= DOUBLED_PAWN_PENALTY; eg_score -= DOUBLED_PAWN_PENALTY; }
            }

            bool isolated = true;
            if (c > 0 && GET_PAWNS(my_pawns, c - 1) > 0) isolated = false;
            if (c < 7 && GET_PAWNS(my_pawns, c + 1) > 0) isolated = false;
            if (isolated) {
                if (is_white) { mg_score += ISOLATED_PAWN_PENALTY; eg_score += ISOLATED_PAWN_PENALTY; }
                else { mg_score -= ISOLATED_PAWN_PENALTY; eg_score -= ISOLATED_PAWN_PENALTY; }
            }

            bool passed = false;
            if (GET_PAWNS(opp_pawns, c) == 0 && (c == 0 || GET_PAWNS(opp_pawns, c - 1) == 0) && (c == 7 || GET_PAWNS(opp_pawns, c + 1) == 0)) {
                passed = true;
            }
            if (passed) {
                float advance = is_white ? (6.0f - (float)r) : ((float)r - 1.0f);
                float mg_bonus = (PASSED_PAWN_BONUS_MG * advance) / 5.0f;
                float eg_bonus = (PASSED_PAWN_BONUS_EG * advance) / 5.0f;
                if (is_white) { mg_score += mg_bonus; eg_score += eg_bonus; }
                else { mg_score -= mg_bonus; eg_score -= eg_bonus; }
            }
        }
        else if (abs_p == 6.0f && phase > 10) {
            if (c <= 2 || c >= 5) {
                int shield_pawns = 0;
                int forward_dir = is_white ? -1 : 1;
                for (int dc = -1; dc <= 1; dc++) {
                    int sc = c + dc;
                    if (sc >= 0 && sc < 8) {
                        int sr = r + forward_dir;
                        if (sr >= 0 && sr < 8 && child_boards[idx * 64 + (sr * 8 + sc)] == (is_white ? 1.0f : -1.0f)) {
                            shield_pawns++;
                        }
                    }
                }
                if (shield_pawns == 0) {
                    if (is_white) mg_score += KING_SHIELD_PENALTY;
                    else mg_score -= KING_SHIELD_PENALTY;
                }
            }
        }
    }

    if (w_bishops >= 2) { mg_score += BISHOP_PAIR_BONUS; eg_score += BISHOP_PAIR_BONUS; }
    if (b_bishops >= 2) { mg_score -= BISHOP_PAIR_BONUS; eg_score -= BISHOP_PAIR_BONUS; }

    if (phase > 24) phase = 24;
    float phase_weight = (float)phase / 24.0f;
    float final_score = (mg_score * phase_weight) + (eg_score * (1.0f - phase_weight));

    if (phase <= 6) {
        if (w_mat > b_mat + 200.0f && w_k_sq != -1 && b_k_sq != -1) {
            final_score += mop_up(w_k_sq, b_k_sq, 1.0f);
        } else if (b_mat > w_mat + 200.0f && w_k_sq != -1 && b_k_sq != -1) {
            final_score += mop_up(b_k_sq, w_k_sq, -1.0f);
        }
    }

    float my_sign = (current_turn == 0.0f) ? 1.0f : -1.0f;
    float final_score_signed = final_score * my_sign;

    scores[idx] = final_score_signed;

    atomicMaxFloat(&global_best[0], final_score_signed);
}