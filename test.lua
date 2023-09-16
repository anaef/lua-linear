local linear = require("linear")

-- Epsilon
local EPSILON = 6E-5

-- Tests the vector function
local function testVector ()
	local X = linear.vector(2)
	assert(#X == 2)
	X[1], X[2] = 1, 2
	assert(X[1] == 1)
	assert(X[2] == 2)
	for i, x in ipairs(X) do
		assert(i == x)
	end
end

-- Tests the matrix function
local function testMatrix ()
	local A = linear.matrix(2, 3)
	assert(#A == 2)
	local A1, A2 = A[1], A[2]
	assert(#A1 == 3)
	A1[1], A1[2], A1[3] = 1, 2, 3
	A2[1], A2[2], A2[3] = 4, 5, 6
	assert(A1[1] == 1)
	assert(A1[2] == 2)
	assert(A1[3] == 3)
	assert(A2[1] == 4)
	assert(A2[2] == 5)
	assert(A2[3] == 6)
	local B = linear.matrix(2, 3, "col")
	assert(#B == 3)
	for i, a in ipairs(A) do
		assert(type(i) == "number")
		assert(linear.type(a) == "vector")
		assert(#a == 3)
	end
end

-- Tests the type function
local function testType ()
	local x = linear.vector(1)
	assert(linear.type(x) == "vector")
	local X = linear.matrix(1, 1)
	assert(linear.type(X) == "matrix")
	assert(linear.type(0) == nil)
end

-- Tests the size function
local function testSize ()
	local x = linear.vector(1)
	assert(linear.size(x) == 1)
	local X = linear.matrix(1, 2)
	local rows, cols, order = linear.size(X)
	assert(rows == 1)
	assert(cols == 2)
	assert(order == "row")
end


-- Tests the tvector function
local function testTvector ()
	local A = linear.matrix(2, 3)
	local A1, A2 = A[1], A[2]
	A1[1], A1[2], A1[3] = 1, 2, 3
	A2[1], A2[2], A2[3] = 4, 5, 6
	local y = linear.tvector(A, 1)
	assert(#y == 2)
	assert(y[1] == 1)
	assert(y[2] == 4)
end

-- Tests the sub function
local function testSub ()
	local x = linear.vector(2)
	x[2] = 1
	local s = linear.sub(x, 2)
	assert(#s == 1)
	assert(s[1] == 1)

	local X = linear.matrix(2, 3)
	X[2][3] = 1
	local S = linear.sub(X, 2, 3)
	assert(#S == 1)
	assert(#S[1] == 1)
	assert(S[1][1] == 1)
end

-- Tests the unwind function
local function testUnwind ()
	local X = linear.matrix(2, 2)
	linear.set(X, 1)
	local Y = linear.matrix(2, 3)
	linear.set(Y, 2)
	local x = linear.vector(2 * 2 + 2 * 3)
	linear.unwind(X, Y, x)
	assert(x[1] == 1)
	assert(x[4] == 1)
	assert(x[5] == 2)
	assert(x[10] == 2)
end

-- Tests the reshape function
local function testReshape ()
	local x = linear.vector(10)
	for i = 1, #x do
		x[i] = i
	end
	local X = linear.matrix(2, 2)
	local Y = linear.matrix(2, 3)
	linear.reshape(x, X, Y)
	assert(X[1][1] == 1)
	assert(X[2][2] == 4)
	assert(Y[1][1] == 5)
	assert(Y[2][3] == 10)
end

-- Tests the totable function
local function testTotable ()
	-- Vector
	local x = linear.vector(2)
	x[1] = 2
	x[2] = 3
	local t = linear.totable(x)
	assert(t.type == "vector")
	assert(t.length == 2)
	assert(type(t.values) == "table")
	assert(#t.values == 2)
	assert(t.values[1] == 2)
	assert(t.values[2] == 3)

	-- Matrix
	local X = linear.matrix(3, 2)
	X[1][1] = 2
	X[3][2] = 4
	local t = linear.totable(X)
	assert(t.type == "matrix")
	assert(t.rows == 3)
	assert(t.cols == 2)
	assert(t.order == "row")
	assert(type(t.values) == "table")
	assert(#t.values == 3)
	assert(type(t.values[1]) == "table")
	assert(#t.values[1] == 2)
	assert(t.values[1][1] == 2)
	assert(t.values[3][2] == 4)
end

-- Tests the tolinear function
local function testTolinear ()
	-- Vector
	local vector = {
		type = "vector",
		length = 2,
		values = { 2, 3 }
	}
	local x = linear.tolinear(vector)
	assert(type(x) == "userdata")
	assert(linear.type(x) == "vector")
	assert(#x == 2)
	assert(x[1] == 2)
	assert(x[2] == 3)

	-- Matrix
	local matrix = {
		type = "matrix",
		rows = 3,
		cols = 2,
		order = "row",
		values = {
			{ 2, 0 },
			{ 0, 0 },
			{ 0, 3 }
		}
	}
	local X = linear.tolinear(matrix)
	assert(type(X) == "userdata")
	assert(linear.type(X) == "matrix")
	assert(#X == 3)
	assert(linear.type(X[1]) == "vector")
	assert(#X[1] == 2)
	assert(X[1][1] == 2)
	assert(X[3][2] == 3)
end

local function testElementaryFunction (inputs, outputs, f)
	local X = linear.matrix(2, #inputs)
	local x1, x2 = X[1], X[2]
	for i, value in ipairs(inputs) do
		x1[i] = value
		x2[#inputs - i + 1] = value
	end
	f(X)
	for i, output in ipairs(outputs) do
		if output == output then
			assert(math.abs(x1[i] - output) < EPSILON, i)
			assert(math.abs(x2[#inputs - i + 1] - output) < EPSILON, i)
		else
			assert(x1[i] ~= x1[i], i)
			assert(x2[#inputs - i + 1] ~= x2[#inputs - i + 1], i)
		end
	end
end

-- Tests the sign function
local function testSgn ()
	testElementaryFunction({ 2, 0, -1, 0 / 0 }, { 1, 0, -1, 0 / 0 }, linear.sgn)
end

-- Tests the abs function
local function testAbs ()
	testElementaryFunction({ 0, -1, 1, 1.5 }, { 0, 1, 1, 1.5 }, linear.abs)
end

-- Tests the exp function
local function testExp ()
	testElementaryFunction({ 0, math.log(2) }, { 1, 2 }, linear.exp)
end

-- Tests the log function
local function testLog ()
	testElementaryFunction({ 1, math.exp(1) }, { 0, 1 }, linear.log)
end

-- Tests the logistic function
local function testLogistic ()
	testElementaryFunction({ 0, -100, 100 }, { 0.5, 0, 1 }, linear.logistic)
end

-- Tests the tanh function
local function testTanh ()
	testElementaryFunction({ 0, 1 }, { 0, 0.76159 }, linear.tanh)
end

-- Tests the softplus function
local function testSoftplus ()
	testElementaryFunction({ -100, -1, 0, 1, 100 }, { 0, 0.31326, 0.69314, 1.31326, 100 },
			linear.softplus)
end

-- Tests the rectifier function
local function testRectifier ()
	testElementaryFunction({ -0.1, 0.1 }, { 0, 0.1 }, linear.rectifier)
end

-- Tests the set function
local function testSet ()
	local x = linear.vector(3)
	linear.set(x, 1)
	assert(x[1] == 1)
	assert(x[2] == 1)
	assert(x[3] == 1)
	local A = linear.matrix(3, 3)
	linear.set(A, 2)
	for i = 1, #A do
		local a = A[i]
		for j = 1, #a do
			assert(a[i] == 2)
		end
	end
end

-- Tests the uniform function
local function testUniform ()
	local x = linear.vector(3)
	linear.uniform(x)
	for i = 1, #x do
		assert(x[i] >= 0 and x[i] < 1)
	end
	local A = linear.matrix(3, 3)
	linear.uniform(A)
	for i = 1, #A do
		local a = A[i]
		for j = 1, #a do
			assert(a[i] >= 0 and a[i] < 1)
		end
	end
end

-- Tests the normal function
local function testNormal ()
	local x = linear.vector(3)
	linear.normal(x)
	for i = 1, #x do
		assert(type(x[i]) == "number")
	end
	local A = linear.matrix(3, 3)
	linear.normal(A)
	for i = 1, #A do
		local a = A[i]
		for j = 1, #a do
			assert(type(a[j]) == "number")
		end
	end
end

-- Tests the inc function
local function testInc ()
	local x = linear.vector(3)
	linear.inc(x, 1)
	for i = 1, #x do
		assert(x[i] == 1)
	end
	local A = linear.matrix(3, 3)
	linear.inc(A, 2)
	for i = 1, #A do
		local a = A[i]
		for j = 1, #a do
			assert(a[j] == 2)
		end
	end
end

-- Tests the scal function
local function testScal ()
	local x = linear.vector(2)
	x[1], x[2] = 1, 2
	linear.scal(x, 2)
	assert(x[1] == 2)
	assert(x[2] == 4)

	local X = linear.matrix(2, 2)
	X[2][2] = 1
	linear.scal(X, 2)
	assert(X[2][2] == 2)
end

-- Tests the pow function
local function testPow ()
	local x = linear.vector(2)
	x[1], x[2] = 1, 2
	linear.pow(x, 0.5)
	assert(x[1] == 1)
	assert(math.abs(x[2] - math.sqrt(2)) < EPSILON)

	local X = linear.matrix(2, 3)
	X[2][3] = 2
	linear.pow(X, 3)
	assert(X[1][1] == 0)
	assert(X[2][3] == 8)
end

-- Tests the apply function
local function testApply ()
	local function inc (x)
		return x + 1
	end
	local A = linear.matrix(2, 3)
	linear.set(A, 1)
	linear.apply(A, inc)
	assert(A[2][2] == 2)
end

-- Tests the dot function
local function testDot ()
	local x = linear.vector(2)
	x[1], x[2] = 1, 2
	assert(linear.dot(x, x) == 5)
end

-- Tests the nrm2 function
local function testNrm2 ()
	local x = linear.vector(2)
	x[1], x[2] = 1, 2
	assert(math.abs(linear.nrm2(x) - math.sqrt(5)) <= EPSILON)
end

-- Tests the asum function
local function testAsum ()
	local x = linear.vector(2)
	x[1], x[2] = 1, -2
	assert(linear.asum(x) == 3)
end

-- Tests the sum function
local function testSum ()
	local x = linear.vector(2)
	x[1], x[2] = 1, -2
	assert(linear.sum(x) == -1)
	local X = linear.matrix(2, 4)
	linear.set(X[1], 1)
	for i = 1, 4 do
		X[2][i] = i
	end
	local x = linear.vector(4)
	linear.sum(X, x)
	assert(x[1] == 2)
	assert(x[2] == 3)
	assert(x[3] == 4)
	assert(x[4] == 5)
	local x = linear.vector(2)
	linear.sum(X, x, "col")
	assert(x[1] == 4)
	assert(x[2] == 10)

	local X = linear.matrix(2, 4, "col")
	linear.set(linear.tvector(X, 1), 1)
	for i = 1, 4 do
		X[i][2] = i
	end
	local x = linear.vector(4)
	linear.sum(X, x, "row")
	assert(x[1] == 2)
	assert(x[2] == 3)
	assert(x[3] == 4)
	assert(x[4] == 5)
	local x = linear.vector(2)
	linear.sum(X, x, "col")
	assert(x[1] == 4)
	assert(x[2] == 10)
end

-- Tests the mean function
local function testMean ()
	local x = linear.vector(3)
	for i = 1, 3 do
		x[i] = 2 * i
	end
	assert(linear.mean(x) == 4)
end

-- Tests the var function
local function testVar ()
	local x = linear.vector(3)
	for i = 1, 3 do
		x[i] = i
	end
	assert(math.abs(linear.var(x) - 2 / 3) < EPSILON)
	assert(math.abs(linear.var(x, 1) - 1) < EPSILON)
	assert(math.abs(linear.var(x, 2) - 2) < EPSILON)
end

-- Tests the std function
local function testStd ()
	local x = linear.vector(3)
	for i = 1, 3 do
		x[i] = i
	end
	assert(math.abs(linear.std(x) - math.sqrt(2 / 3)) < EPSILON)
	assert(math.abs(linear.std(x, 1) - 1) < EPSILON)
	assert(math.abs(linear.std(x, 2) - math.sqrt(2)) < EPSILON)
end

-- Tests the iamax function
local function testIamax ()
	local x = linear.vector(2)
	x[1], x[2] = 1, -2
	assert(linear.iamax(x) == 2)
end

-- Tests the iamin function
local function testIamin ()
	local x = linear.vector(2)
	x[1], x[2] = 1, -2
	assert(linear.iamin(x) == 1)
end

-- Tests the swap function
local function testSwap ()
	local x, y = linear.vector(2), linear.vector(2)
	x[1], y[2] = 1, 2
	linear.swap(x, y)
	assert(x[1] == 0)
	assert(x[2] == 2)
	assert(y[1] == 1)
	assert(y[2] == 0)

	local X = linear.matrix(2, 3)
	local Y = linear.matrix(2, 3)
	X[2][3], Y[2][3] = 1, 2
	linear.swap(X, Y)
	assert(X[2][3] == 2)
	assert(Y[2][3] == 1)
end

-- Tests the copy function
local function testCopy ()
	local x = linear.vector(2)
	x[1], x[2] = 1, 2
	local y = linear.vector(2)
	linear.copy(x, y)
	assert(y[1] == 1)
	assert(y[2] == 2)

	local X = linear.matrix(2, 3)
	local Y = linear.matrix(2, 3)
	X[2][3] = 1
	linear.copy(X, Y)
	assert(Y[2][3] == 1)
end

-- Tests the axpy function
local function testAxpy ()
	local x = linear.vector(2)
	x[1], x[2] = 1, 2
	local y = linear.vector(2)
	y[1], y[2] = 3, 4
	linear.axpy(x, y, 2)
	assert(y[1] == 5)
	assert(y[2] == 8)

	local Y = linear.matrix(2, 3)
	linear.set(Y[1], 1)
	linear.set(Y[2], 2)
	local x = linear.vector(3)
	for i = 1, 3 do
		x[i] = i
	end
	linear.axpy(x, Y)
	assert(Y[1][1] == 2)
	assert(Y[1][2] == 3)
	assert(Y[1][3] == 4)
	assert(Y[2][1] == 3)
	assert(Y[2][2] == 4)
	assert(Y[2][3] == 5)
	linear.set(Y[1], 1)
	linear.set(Y[2], 2)
	local x = linear.vector(2)
	for i = 1, 2 do
		x[i] = i
	end
	linear.axpy(x, Y, 2, "col")
	assert(Y[1][1] == 3)
	assert(Y[1][2] == 3)
	assert(Y[1][3] == 3)
	assert(Y[2][1] == 6)
	assert(Y[2][2] == 6)
	assert(Y[2][3] == 6)

	local Y = linear.matrix(2, 3, "col")
	linear.set(linear.tvector(Y, 1), 1)
	linear.set(linear.tvector(Y, 2), 2)
	local x = linear.vector(3)
	for i = 1, 3 do
		x[i] = i
	end
	linear.axpy(x, Y, 1, "row")
	assert(Y[1][1] == 2)
	assert(Y[2][1] == 3)
	assert(Y[3][1] == 4)
	assert(Y[1][2] == 3)
	assert(Y[2][2] == 4)
	assert(Y[3][2] == 5)
	linear.set(linear.tvector(Y, 1), 1)
	linear.set(linear.tvector(Y, 2), 2)
	local x = linear.vector(2)
	for i = 1, 2 do
		x[i] = i
	end
	linear.axpy(x, Y, 3, "col")
	assert(Y[1][1] == 4)
	assert(Y[2][1] == 4)
	assert(Y[3][1] == 4)
	assert(Y[1][2] == 8)
	assert(Y[2][2] == 8)
	assert(Y[3][2] == 8)

	local X = linear.matrix(2, 2)
	local Y = linear.matrix(2, 2)
	X[2][2] = 1
	Y[2][2] = 2
	linear.axpy(X, Y, 2)
	assert(Y[2][2] == 4)

end

-- Tests the axpby function
local function testAxpby ()
	local x = linear.vector(2)
	x[1], x[2] = 1, 2
	local y = linear.vector(2)
	y[1], y[2] = 3, 4
	linear.axpby(x, y, 2, 3)
	assert(y[1] == 11)
	assert(y[2] == 16)

	local X = linear.matrix(2, 2)
	local Y = linear.matrix(2, 2)
	X[2][2] = 1
	Y[2][2] = 2
	linear.axpby(X, Y, 2, 3)
	assert(Y[2][2] == 8)

	linear.axpby(x, Y)
	assert(Y[1][1] == 1)
	assert(Y[1][2] == 2)
	assert(Y[2][1] == 1)
	assert(Y[2][2] == 2)
end

-- Tests the mul function
local function testMul ()
	local x = linear.vector(2)
	x[1], x[2] = 1, 2
	linear.mul(x, x)
	assert(x[1] == 1)
	assert(x[2] == 4)
	linear.mul(x, x, 0)
	assert(x[1] == 1)
	assert(x[2] == 4)
	linear.mul(x, x, -1)
	assert(x[1] == 1)
	assert(x[2] == 1)

	local X = linear.matrix(2, 2)
	X[1][1] = 1
	X[1][2] = 2
	X[2][1] = 2
	X[2][2] = 4
	linear.mul(X, X)
	assert(X[1][1] == 1)
	assert(X[1][2] == 4)
	assert(X[2][1] == 4)
	assert(X[2][2] == 16)
	x[2] = 2
	linear.mul(x, X, -1)
	assert(X[1][1] == 1)
	assert(X[1][2] == 2)
	assert(X[2][1] == 4)
	assert(X[2][2] == 8)
	linear.mul(x, X, 0.5)
	assert(X[1][1] == 1)
	assert(math.abs(X[1][2] - 2 * math.sqrt(2)) < EPSILON)
	assert(X[2][1] == 4)
	assert(math.abs(X[2][2] - 8 * math.sqrt(2)) < EPSILON)
end

-- Tests the gemv function
local function testGemv ()
	local A = linear.matrix(2, 3)
	local A1, A2 = A[1], A[2]
	A1[1], A1[2], A1[3] = 1, 2, 3
	A2[1], A2[2], A2[3] = 4, 5, 6
	local x = linear.vector(3)
	x[1], x[2], x[3] = 1, 2, 3
	local y = linear.vector(2)
	linear.gemv(A, x, y)
	assert(y[1] == 14)
	assert(y[2] == 32)

	local A = linear.matrix(3, 2)
	local A1, A2, A3 = A[1], A[2], A[3]
	A1[1], A1[2] = 1, 4
	A2[1], A2[2] = 2, 5
	A3[1], A3[2] = 3, 6
	linear.gemv(A, x, y, 2, nil, "trans")
	assert(y[1] == 28)
	assert(y[2] == 64)
end

-- Tests the ger function
local function testGer ()
	local x, y = linear.vector(2), linear.vector(3)
	local A = linear.matrix(2, 3)
	x[1], x[2] = 1, 2
	y[1], y[2], y[3] = 1, 2, 3
	linear.ger(x, y, A)
	for i = 1, #x do
		for j = 1, #y do
			assert(A[i][j] == x[i] * y[j])
		end
	end
end

-- Tests the gemm function
local function testGemm ()
	local A = linear.matrix(2, 3)
	local A1, A2 = A[1], A[2]
	A1[1], A1[2], A1[3] = 1, 2, 3
	A2[1], A2[2], A2[3] = 4, 5, 6
	local B = linear.matrix(3, 2)
	local B1, B2, B3 = B[1], B[2], B[3]
	B1[1], B1[2] = 1, 2
	B2[1], B2[2] = 3, 4
	B3[1], B3[2] = 5, 6
	local C = linear.matrix(2, 2)
	local C1, C2 = C[1], C[2]
	linear.gemm(A, B, C)
	assert(C1[1] == 22)
	assert(C1[2] == 28)
	assert(C2[1] == 49)
	assert(C2[2] == 64)

	local C = linear.matrix(3, 3)
	local C1, C2, C3 = C[1], C[2], C[3]
	linear.gemm(A, B, C, 2, nil, "trans", "trans")
	assert(C1[1] == 18)
	assert(C1[2] == 38)
	assert(C1[3] == 58)
	assert(C2[1] == 24)
	assert(C2[2] == 52)
	assert(C2[3] == 80)
	assert(C3[1] == 30)
	assert(C3[2] == 66)
	assert(C3[3] == 102)
end

-- Tests the gesv function
local function testGesv ()
	local A = linear.matrix(3, 3)
	local A1, A2, A3 = A[1], A[2], A[3]
	A1[1], A1[2], A1[3] = 8, 1, 6
	A2[1], A2[2], A2[3] = 3, 5, 7
	A3[1], A3[2], A3[3] = 4, 9, 2
	local B = linear.matrix(3, 1)
	linear.set(B, 1)
	linear.gesv(A, B)
	local b = linear.tvector(B, 1)
	assert(math.abs(b[1] - 0.066667) < 0.001)
	assert(math.abs(b[2] - 0.066667) < 0.001)
	assert(math.abs(b[3] - 0.066667) < 0.001)
end

-- Tests the gels function
local function testGels ()
	local A = linear.matrix(3, 2)
	local A1, A2, A3 = A[1], A[2], A[3]
	A1[1], A1[2] = 1, 2
	A2[1], A2[2] = 3, 4
	A3[1], A3[2] = 5, 6
	local B = linear.matrix(3, 1)
	B[1][1], B[2][1], B[3][1] = 1, 2, 3
	linear.gels(A, B)
	assert(math.abs(B[1][1] - 0) < EPSILON)
	assert(math.abs(B[2][1] - 0.5) < EPSILON)
	assert(math.abs(B[3][1] - 0) < EPSILON)
end

-- Tests the inv function
local function testInv ()
	local A = linear.matrix(3, 3)
	local A1, A2, A3 = A[1], A[2], A[3]
	A1[1], A1[2], A1[3] = 8, 1, 6
	A2[1], A2[2], A2[3] = 3, 5, 7
	A3[1], A3[2], A3[3] = 4, 9, 2
	assert(linear.inv(A) == 0)
	assert(math.abs(A1[1] - 0.147222) < 0.0001)
	assert(math.abs(A1[2] + 0.144444) < 0.0001)
	assert(math.abs(A1[3] - 0.063889) < 0.0001)
	assert(math.abs(A2[1] + 0.061111) < 0.0001)
	assert(math.abs(A2[2] - 0.022222) < 0.0001)
	assert(math.abs(A2[3] - 0.105556) < 0.0001)
	assert(math.abs(A3[1] + 0.019444) < 0.0001)
	assert(math.abs(A3[2] - 0.188889) < 0.0001)
	assert(math.abs(A3[3] + 0.102778) < 0.0001)
end

-- Tests the det function
local function testDet ()
	local A = linear.matrix(3, 3)
	local A1, A2, A3 = A[1], A[2], A[3]
	A1[1], A1[2], A1[3] = 8, 1, 6
	A2[1], A2[2], A2[3] = 3, 5, 7
	A3[1], A3[2], A3[3] = 4, 9, 2
	assert(math.abs(linear.det(A) - -360) < EPSILON)
	A[1][1] = -8
	assert(math.abs(linear.det(A) - 488) < EPSILON)
end

-- Tests the cov function
local function testCov ()
	local A = linear.matrix(3, 2)
	local A1, A2, A3 = A[1], A[2], A[3]
	A1[1], A1[2] = 1, 1
	A2[1], A2[2] = 1, 2
	A3[1], A3[2] = 2, 2
	local B = linear.matrix(2, 2)
	linear.cov(A, B, 1)
	assert(math.abs(B[1][1] - 1 / 3) < EPSILON)
	assert(math.abs(B[1][2] - 1 / 6) < EPSILON)
	assert(math.abs(B[2][1] - 1 / 6) < EPSILON)
	assert(math.abs(B[2][2] - 1 / 3) < EPSILON)
end

-- Tests the corr function
local function testCorr ()
	local A = linear.matrix(3, 2)
	local A1, A2, A3 = A[1], A[2], A[3]
	A1[1], A1[2] = 1, 1
	A2[1], A2[2] = 1, 2
	A3[1], A3[2] = 2, 2
	local B = linear.matrix(2, 2)
	linear.corr(A, B, 1)
	assert(math.abs(B[1][1] - 1) < EPSILON)
	assert(math.abs(B[1][2] - 0.5) < EPSILON)
	assert(math.abs(B[2][1] - 0.5) < EPSILON)
	assert(math.abs(B[2][2] - 1) < EPSILON)
end

-- Structural function tests
testVector()
testMatrix()
testType()
testSize()
testTvector()
testSub()
testUnwind()
testReshape()
testTotable()
testTolinear()

-- Elementary function tests
testSgn()
testAbs()
testExp()
testLog()
testLogistic()
testTanh()
testSoftplus()
testRectifier()
testSet()
testUniform()
testNormal()
testInc()
testScal()
testPow()
testApply()

-- Vector function tests
testDot()
testNrm2()
testAsum()
testSum()
testMean()
testVar()
testStd()
testIamax()
testIamin()

-- Vector-matrix function tests
testSwap()
testCopy()
testAxpy()
testAxpby()
testMul()

-- Matrix function tests
testGemv()
testGer()
testGemm()
testGesv()
testGels()
testInv()
testDet()

-- Statistical function tests
testCov()
testCorr()

-- Exit
os.exit(0)
