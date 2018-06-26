#include <iostream>
#include <matrix.h>
#include "mex.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"

using namespace Eigen;
using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
	double *imagepadding, *H_16, *parameters, *dt;
	int pad_rows, pad_cols, dt_rows, dt_cols, had_rows, had_cols;
	
	const int scale = 2;
	const int offset = scale/2;


	RowVectorXi H_15(15);
	H_15 << 2,8,3,12,10,1,4,11,14,6,9,15,7,13,5;

	mxAssert(nlhs==1 && nrhs==4,"Error: number of variables");

	pad_rows = mxGetM( prhs[0] );  // imagepadding
	pad_cols = mxGetN( prhs[0] );

	const int *Para_dim;
	Para_dim = (const int*)mxGetDimensions( prhs[1] );  // the dimension of the mapping models 

	dt_rows = mxGetM( prhs[2] );   // the learned decision tree
	dt_cols = mxGetN( prhs[2] );

	had_rows = mxGetM( prhs[3] );  // Hadamard Matrix
	had_cols = mxGetN( prhs[3] );

	imagepadding = mxGetPr( prhs[0] );   // LR input   
	parameters = mxGetPr(prhs[1]);     // the parameters - the learned mapping models
	dt = mxGetPr( prhs[2] );   //  the learned decision tree, it is used to find the approprate mapping model.
	H_16 = mxGetPr( prhs[3] );  // Hadamard Matrix

	Map< MatrixXd > LR_INPUT ( imagepadding, pad_rows, pad_cols );  // lr_input
	Map< MatrixXd > DECISION_TREE( dt, dt_rows, dt_cols);  // Dt
	Map< MatrixXd > Hadamard_15( H_16, had_rows, had_cols );  // h_16
	
	MatrixXd LRblock(4,4);
	RowVectorXd pattern(15), HRB(scale*scale);

	MatrixXd imageH( (pad_rows - 2) * scale , (pad_cols - 2) * scale  );
	imageH << MatrixXd::Zero( (pad_rows - 2) * scale , (pad_cols - 2) * scale  ); 

	for ( int ii=1; ii < LR_INPUT.rows() - 4; ++ii )
	{
		for ( int jj=1; jj < LR_INPUT.cols() - 4; ++jj )
		{
			LRblock = LR_INPUT.block<4,4>( ii, jj );

			Map< RowVectorXd > LRB( LRblock.data(), LRblock.size() );
			
			pattern = LRB * Hadamard_15;

			int ptr = 0;
			int m = DECISION_TREE( ptr, 0 );
			
			while ( m != 0 )
			{
				double val = pattern( H_15(m-1)-1 );
				if ( val < DECISION_TREE(ptr,4) )
					ptr = DECISION_TREE(ptr,1)-1;
				else 
				{
					if ( val > DECISION_TREE(ptr,5) )
						ptr = DECISION_TREE(ptr,3)-1;
					else
						ptr = DECISION_TREE(ptr,2)-1;
				}
				m = DECISION_TREE(ptr,0);
			}

			int num = DECISION_TREE(ptr,1);

			Map< MatrixXd > para( parameters + ( (num - 1) * (*Para_dim) * (*(Para_dim + 1)) ), *Para_dim, *(Para_dim + 1));
			
			HRB = para  * LRB.transpose();
			
			Map< MatrixXd > lala( HRB.data(), scale , scale );
			imageH.block< scale , scale  >( ii * scale  + offset , jj * scale  + offset ) = lala;

		}
	}

	plhs[0] = mxCreateDoubleMatrix( (pad_rows - 2) * scale , (pad_cols - 2) * scale , mxREAL); 
	double *output = mxGetPr(plhs[0]);
	for (int i = 0; i < ((pad_rows - 2) * scale ); ++i )
	{
		for ( int j = 0; j < ((pad_cols - 2) * scale ); ++j )
		{
			output[ j * ((pad_rows - 2) * scale ) + i ] = imageH(i,j);
		}
	}

}