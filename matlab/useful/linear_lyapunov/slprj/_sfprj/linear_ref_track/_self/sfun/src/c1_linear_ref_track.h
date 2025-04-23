#ifndef __c1_linear_ref_track_h__
#define __c1_linear_ref_track_h__

/* Type Definitions */
#ifndef typedef_SFc1_linear_ref_trackInstanceStruct
#define typedef_SFc1_linear_ref_trackInstanceStruct

typedef struct {
  SimStruct *S;
  ChartInfoStruct chartInfo;
  uint32_T chartNumber;
  uint32_T instanceNumber;
  int32_T c1_sfEvent;
  boolean_T c1_doneDoubleBufferReInit;
  uint8_T c1_is_active_c1_linear_ref_track;
  void *c1_fEmlrtCtx;
  real_T *c1_L;
  real_T (*c1_x_des)[3];
  real_T *c1_u;
  real_T (*c1_x)[2];
  real_T (*c1_x_d)[2];
  real_T *c1_eps;
  real_T *c1_k_out;
} SFc1_linear_ref_trackInstanceStruct;

#endif                                 /*typedef_SFc1_linear_ref_trackInstanceStruct*/

/* Named Constants */

/* Variable Declarations */
extern struct SfDebugInstanceStruct *sfGlobalDebugInstanceStruct;

/* Variable Definitions */

/* Function Declarations */
extern const mxArray *sf_c1_linear_ref_track_get_eml_resolved_functions_info
  (void);

/* Function Definitions */
extern void sf_c1_linear_ref_track_get_check_sum(mxArray *plhs[]);
extern void c1_linear_ref_track_method_dispatcher(SimStruct *S, int_T method,
  void *data);

#endif
