classdef Test_BPMF < matlab.unittest.TestCase
	%% TEST_BPMF 

	%  Usage:  >> results = run(mlsalakhudtinov_unittest.Test_BPMF)
 	%          >> result  = run(mlsalakhudtinov_unittest.Test_BPMF, 'test_dt')
 	%  See also:  file:///Applications/Developer/MATLAB_R2014b.app/help/matlab/matlab-unit-test-framework.html

	%  $Revision$
 	%  was created 01-Oct-2017 23:50:36 by jjlee,
 	%  last modified $LastChangedDate$ and placed into repository /Users/jjlee/Local/src/mlcvl/mlsalakhudtinov/test/+mlsalakhudtinov_unittest.
 	%% It was developed on Matlab 9.3.0.713579 (R2017b) for MACI64.  Copyright 2017 John Joowon Lee.
 	
	properties
 		registry
 		testObj
 	end

	methods (Test)
		function test_afun(this)
 			import mlsalakhudtinov.*;
 			this.assumeEqual(1,1);
 			this.verifyEqual(1,1);
 			this.assertEqual(1,1);
 		end
	end

 	methods (TestClassSetup)
		function setupBPMF(this)
 			import mlsalakhudtinov.*;
 			this.testObj_ = BPMF;
 		end
	end

 	methods (TestMethodSetup)
		function setupBPMFTest(this)
 			this.testObj = this.testObj_;
 			this.addTeardown(@this.cleanFiles);
 		end
	end

	properties (Access = private)
 		testObj_
 	end

	methods (Access = private)
		function cleanFiles(this)
 		end
	end

	%  Created with Newcl by John J. Lee after newfcn by Frank Gonzalez-Morphy
 end

