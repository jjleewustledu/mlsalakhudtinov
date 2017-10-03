classdef Test_DBM < matlab.unittest.TestCase
	%% TEST_DBM 

	%  Usage:  >> results = run(mlsalakhudtinov_unittest.Test_DBM)
 	%          >> result  = run(mlsalakhudtinov_unittest.Test_DBM, 'test_dt')
 	%  See also:  file:///Applications/Developer/MATLAB_R2014b.app/help/matlab/matlab-unit-test-framework.html

	%  $Revision$
 	%  was created 01-Oct-2017 23:51:02 by jjlee,
 	%  last modified $LastChangedDate$ and placed into repository /Users/jjlee/Local/src/mlcvl/mlsalakhudtinov/test/+mlsalakhudtinov_unittest.
 	%% It was developed on Matlab 9.3.0.713579 (R2017b) for MACI64.  Copyright 2017 John Joowon Lee.
 	
	properties
        pwd0
 		registry
 		testObj
 	end

	methods (Test)
        function test_eval(this)
            a = 'letter A'; b = 'letter B'; %#ok<NASGU>
            that = eval('this.feval(a, b)');
            this.assertClass(that, 'mlsalakhudtinov_unittest.Test_DBM');
        end
		function test_demo_small(this)
 			import mlsalakhudtinov.*;
            this.testObj = this.testObj.demo_small;            
 			%this.assumeEqual(1,1);
 			%this.verifyEqual(this, []);
 			%this.assertEqual(1,1);
        end
        function test_converter(this)
            this.testObj = this.testObj.converter;            
            for d = 0:9
                this.assertTrue(lexist(['digit' num2str(d) '.mat'], 'file'));
            end
        end
	end

 	methods (TestClassSetup)
		function setupDBM(this)
 			import mlsalakhudtinov.*;
 			this.testObj_ = DBM;
            this.pwd0 = pushd(fullfile(getenv('LOCAL'), 'src', 'mlcvl', 'mlsalakhudtinov', 'test', '+mlsalakhudtinov_unittest'));
 			this.addTeardown(@this.cleanFilesystem);
            rng('default');
 		end
	end

 	methods (TestMethodSetup)
		function setupDBMTest(this)
 			this.testObj = this.testObj_;
            warning('off','MATLAB:RandStream:ActivatingLegacyGenerators');
            warning('off','MATLAB:RandStream:ReadingInactiveLegacyGeneratorState');
 		end
	end

	properties (Access = private)
 		testObj_
 	end

	methods (Access = private)
		function cleanFilesystem(this)
            popd(this.pwd0);
            warning('on','MATLAB:RandStream:ActivatingLegacyGenerators');
            warning('on','MATLAB:RandStream:ReadingInactiveLegacyGeneratorState');
        end
        function this = feval(this, arg, arg2)
            assert(ischar(arg));
            assert(ischar(arg2));
            fprintf(['feval received ' arg ', ' arg2 '\n']);
        end
	end

	%  Created with Newcl by John J. Lee after newfcn by Frank Gonzalez-Morphy
 end

