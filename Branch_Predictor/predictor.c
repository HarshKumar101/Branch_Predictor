//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include "predictor.h"


const char *studentName = "XYZ";
const char *studentID   = "XYZ";
const char *email       = "XYZ@XYZ.com";


const char *bpName[4] = { "Static", "Gshare",
                          "Tournament", "Custom" };


int ghistoryBits; // Number of bits used for Global History
int lhistoryBits; // Number of bits used for Local History
int pcIndexBits;  // Number of bits used for PC index
int bpType;       // Branch Prediction Type
int verbose;

// Variables for Gshare and Tournament
uint32_t* pht_table;
uint8_t*  bht_table;
uint16_t  GHR;
uint8_t*  gshare_bht_table;
uint8_t*  choice_table;

// Variables for Custom
int      ghr_len;
uint32_t ghr_reg;
int      design_para;
int      theta;
int8_t   **wght;
int      sum;


/*----------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------ MODEL INITIALIZATION ------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------*/

void gshare_init(){
    GHR = 0;
    gshare_bht_table = (uint8_t *)malloc((1<<ghistoryBits) * sizeof(uint8_t));
    for(int i = 0; i < (1<<ghistoryBits); i++)
        gshare_bht_table[i] = 1;
}


void local_init(){
    pht_table =  (uint32_t *) malloc((1<<pcIndexBits) * sizeof(uint32_t));
    bht_table =  (uint8_t *)  malloc((1<<lhistoryBits)* sizeof(uint8_t));
    for(int i = 0; i < (1<<pcIndexBits); i++)
        pht_table[i] = 0;

    for(int i = 0; i < (1<<lhistoryBits); i++)
        bht_table[i] = 1;
}


void tournament_init(){
    gshare_init();
    local_init();
    choice_table = (uint8_t *)malloc((1<<ghistoryBits) * sizeof(uint8_t));
    for(int i = 0; i < (1<<ghistoryBits); i++)
        choice_table[i] = 1;
}

/* Implemented Neural Network based Branch Predictor by referring to the below paper
   https://www.microarch.org/micro36/html/pdf/jimenez-FastPath.pdf */
void custom_init()
{
    ghr_len     = 13;
    design_para = 195;
    ghr_reg     = 0;
    theta       = (1.0*ghr_len + 5);

    /*
    /// Memory Allocation to Weights ////

        Weights is a double matrix;
        No. of Rows = design_para
        No. of Cols = ghr_len+1
        Each element of the Weight matrix stores value from [-32,31] as coded in the custom_train function (though they are defined as int8_t).
        So each weight is only taking 6 bits instead of 8

        So total no. of bits used = 6*design_para*ghr_len = 6*195*14 = 16380 bits < 16384 (16k bits)
        there are 6 other variables being used. so they need 32*6 = 192 bits
    */

    wght = malloc(sizeof(int8_t*)*design_para);
    for(int i = 0;i<design_para;i++)
        wght[i] = malloc(sizeof(int8_t*)*(ghr_len+1));

    // Initialization of Weights to 0
    for(int i = 0;i<design_para;i++){
        for(int j = 0;j<ghr_len+1;j++){
            wght[i][j] = 2;
        }
    }

}


void
init_predictor()
{
    switch (bpType) {
    case STATIC:
        break;
    case GSHARE:
        gshare_init();
        break;
    case TOURNAMENT:
        tournament_init();
        break;
    case CUSTOM:
        custom_init();
        break;
    default:
        break;
  }

}

/*----------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------ MODEL PREDICTIOM ----------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------*/

uint8_t gshare_pred(uint32_t pc){
    int ghr_bht_index = (pc & ((1<<ghistoryBits) - 1 ) ) ^ (GHR& ((1<<ghistoryBits) - 1));
    int ghr_bht_val   = gshare_bht_table[ghr_bht_index] & 3;
    if (ghr_bht_val == 0 || ghr_bht_val == 1)
        return 0;
    else
        return 1;
}


uint8_t local_pred(uint32_t pc){
    int pht_index = pc&((1<<pcIndexBits)-1);
    int bht_index = (pht_table[pht_index]) & ((1<<lhistoryBits) - 1);
    int a = bht_table[bht_index] & 3;
    if (a == 0 || a == 1)
        return 0;
    else
        return 1;
}


uint8_t tournament_pred(uint32_t pc){
    if(choice_table[GHR & ((1<<ghistoryBits) - 1)] <2)
        return gshare_pred(0);
    else
        return local_pred(pc);
}


uint8_t custom_pred(uint32_t pc)
{
    int index = pc % design_para;
    sum = wght[index][0];
    for(int i = 1;i<ghr_len+1;i++){
        int a = (ghr_reg>>(i-1)) &1;
        if(a == 1)
            sum+= wght[index][i];
        else
            sum-= wght[index][i];
    }
    if (sum >= 0) return 1;
    else return 0;
}


uint8_t
make_prediction(uint32_t pc)
{
  switch (bpType) {
    case STATIC:
        return TAKEN;
        break;
    case GSHARE:
        return gshare_pred(pc);
        break;
    case TOURNAMENT:
        return tournament_pred(pc);
        break;
    case CUSTOM:
        return custom_pred(pc);
        break;
    default:
      break;
  }

  return NOTTAKEN;
}

/*----------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------- MODEL TRAINING ----------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------*/
void gshare_train(uint32_t pc, uint8_t outcome){
    int ghr_bht_index = (pc & ((1<<ghistoryBits) - 1) ) ^ (GHR & ((1<<ghistoryBits)-1));
    int ghr_bht_val   = gshare_bht_table[ghr_bht_index] & 3;

    if(outcome == 1){
        if(ghr_bht_val != 3)
            gshare_bht_table[ghr_bht_index] = (gshare_bht_table[ghr_bht_index]&3) + 1;
    }
    if(outcome == 0){
        if(ghr_bht_val != 0)
            gshare_bht_table[ghr_bht_index] = (gshare_bht_table[ghr_bht_index]&3) - 1;
    }
    GHR = (GHR<<1) | outcome;
}


void local_train(uint32_t pc, uint8_t outcome){
    int pht_index = pc&((1<<pcIndexBits)-1);
    int bht_index = pht_table[pht_index]&((1<<lhistoryBits)-1);
    int a = bht_table[bht_index] & 3;
    pht_table[pht_index] = ((pht_table[pht_index]<<1) | outcome) & ((1<<lhistoryBits) - 1);

    if(outcome == 1){
        if(a != 3)
            bht_table[bht_index] = (bht_table[bht_index] & 3)+1;
    }
    if (outcome == 0){
        if(a != 0)
            bht_table[bht_index] = (bht_table[bht_index] & 3)-1;
    }
}


void tournament_train(uint32_t pc, uint8_t outcome){
    if(gshare_pred(0) != local_pred(pc)){
        int choice_index = GHR & ((1<<ghistoryBits)-1);
        if(outcome == gshare_pred(0)){
            if(choice_table[choice_index] !=0)
                choice_table[choice_index] = choice_table[choice_index] - 1;
        }
        if(outcome == local_pred(pc)){
            if(choice_table[choice_index] !=3)
                choice_table[choice_index] = choice_table[choice_index] + 1;
        }

    }

    gshare_train(0,outcome);
    local_train(pc,outcome);
}


void custom_train(uint32_t pc, uint8_t outcome)
{
    int index = pc % design_para;
    if(outcome !=custom_pred(pc) || (sum <= theta && sum >= -theta)){
        if(outcome == 1&& wght[index][0]!= 31)
            wght[index][0] += 1;
        else if (wght[index][0]!= -32)
            wght[index][0] -= 1;

        for(int i=1;i<ghr_len+1;i++){
            int a = (ghr_reg>>(i-1)) &1;
            if(a == outcome && wght[index][i]!= 31)
                wght[index][i]+=1;
            else if (wght[index][i]!= -32)
                wght[index][i]-=1;
        }
    }
    ghr_reg = ((ghr_reg<<1) | outcome) & ((1<<ghr_len)-1);
}


void
train_predictor(uint32_t pc, uint8_t outcome)
{
    switch (bpType) {
    case STATIC:
        break;
    case GSHARE:
        gshare_train(pc,outcome);
        break;
    case TOURNAMENT:
        tournament_train(pc,outcome);
        break;
    case CUSTOM:
        custom_train(pc, outcome);
        break;
    default:
        break;
  }
}
