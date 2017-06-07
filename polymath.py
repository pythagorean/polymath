from sympy import (
     Matrix, Rational, Symbol, nfloat, nsimplify, gcd_list, ilcm,
     binomial, factorial, zeros, eye, pi)
from numpy import nonzero, array, ndarray, matrix as npmatrix
from decimal import Decimal, getcontext as decimalcontext
from fractions import Fraction
from itertools import count, product, chain
from re import compile

from sympy import Expr
def _to_nfloat(*args): return nfloat(*args)
Expr.to_nfloat = _to_nfloat

# Exact representations of roots and methods for projecting them to line or plane with arbitrary precision,
# hyperdimensional algebras and other techniques to remove irrationality from algebraic numbers.

# This code is not highly optimized and may contain bugs which make it unsuitable for production environments.
# Correct bug fixes and suggestions to improve readability, concision and pythonicity are always most welcome.
# Performance tweaks invited if new bugs avoided and gains are substantial enough to justify any loss of clarity.

def _rationalized(*args):
    newargs = []
    for arg in args:
        if type(arg) in (float, str, int, Decimal, Fraction): newargs.append(nsimplify(arg))
        else: newargs.append(arg)
    return tuple(newargs)

def _complex_split(*args):
    pairs = []
    for arg in args:
        if type(arg) is complex: pairs += list(_rationalized(arg.real, arg.imag))
        elif getattr(arg, 'is_complex', False): pairs += list(arg.as_real_imag())
        else: pairs += list(_rationalized(arg, arg*0))
    return tuple(pairs)

def pow(x, p, precision=16):
    # precision used for matrix power projection
    x,xj,p,pj = _complex_split(x,p)
    if xj: x = Root(eye(2)@x + Matrix(jay(2))@xj)
    if pj: p = Root(eye(2)@p + Matrix(jay(2))@pj)
    if getattr(p, 'is_Matrix', 0):
        return exp(Root(p)@ln(x, precision), precision)
    if p.q > 1:
        x = Matrix([x])
        scale = x.rows
        order = p.q
        root = Root(zeros(order))
        root[:-1,1:] = eye(order-1)
        root = root.enlarge(scale)
        root[scale*(order-1):,:scale] = x
        x = root
    p = p.p
    if p < 0: return 1/pow(x,-p)
    if p == 0: return x/x
    step = 1
    prev = []
    while step + step <= p:
        prev.append((step,x))
        x *= x
        step += step
    while step < p:
        while step + prev[-1][0] > p: prev.pop()
        x *= prev[-1][1]
        step += prev.pop()[0]
    return x
def jay(order=2): return pow(-1, 1/order)
def ell(order=2): return pow( 1, 1/order)

def exp(x, precision=16):
    # precision equivalent to number of nfloat digits
    return Root([pow(x,1)]).exponential_projection(precision)
def ln(x, precision=16):
    x = Root([pow(x,1)])
    logarithm = x.logarithmic_projection(precision)
    if logarithm is not None: return logarithm
    # projection to complex plane requires pi
    x,j = x.to_nfloat(precision), jay().to_nfloat(precision)
    if not x: raise ValueError("Domain error: ln(0) not defined.")
    return nfloat(ln(abs(x), precision) + atan2(*x.as_real_imag()[::-1], precision) * j, precision)
def log(x, base=10, precision=16):
    return (ln(x, precision) / ln(base, precision)).to_nfloat(precision)

def sin(theta, precision=16): return sincos(theta, precision)[0]
def cos(theta, precision=16): return sincos(theta, precision)[1]
def sinh(x, precision=16):    return sinhcosh(x, precision)[0]
def cosh(x, precision=16):    return sinhcosh(x, precision)[1]
def sincos(theta, precision=16):
    sinh,cosh = sinhcosh(jay() @ theta, precision)
    return ~jay() @ sinh, cosh
def sinhcosh(x, precision=16):
    epx,emx = exp(x, precision), exp(-x, precision)
    return (epx-emx)/Rational(2), (epx+emx)/Rational(2)
def tan(theta, precision=16):
    return ~jay() @ tanh(jay() @ theta, precision)
def tanh(x, precision=16):
    epx,emx = exp(x, precision), exp(-x, precision)
    return (epx-emx)/(epx+emx)
def atanh(x, precision=16):
    x = Root([pow(x,1)])
    xf = x.to_nfloat(precision)
    if xf in (-1,1): raise ValueError("Domain error: atanh(%d) not defined." % int(xf))
    return ln((1+x)/(1-x), precision)/Rational(2)
def atan(x, digits=16):
    x = Root([pow(x,1)])
    xf = x.to_nfloat(digits)
    if not xf.is_real:
        jx = jay()@x
        if jx.to_nfloat(digits) in (-1,1):
            raise ValueError("Domain error: atan(%s) not defined." % str(complex(xf)))
        return (jay()@(ln(1-jx)-ln(1+jx))/Rational(2)).to_nfloat(digits)
    if xf < 0: return -atan(-x, digits)
    if xf > 1: return -atan(~x, digits) + atan(1, digits)*2
    def step(x,n): return 2**(n+n-2) * factorial(n-1)**2/factorial(n+n-1) * x**(n+n-1)/(1+x*x)**n
    return Root([xf]).sumseries_projection(step, digits)
def atan2(y, x, digits=16):
    if x == y == 0: raise ValueError("Domain error: atan2(0,0) not defined.")
    if not x: return (y>0 and 2 or -2) * atan(1, digits)
    if x > 0: return atan(y/x, digits)
    return atan(y/x, digits) + (y>=0 and 4 or -4) * atan(1, digits)

class Root(Matrix):
    def enlarge(root, scale):
        # increase root matrix size by scale factor
        oldsize = root.rows
        newsize = oldsize * scale
        newroot = type(root)(zeros(newsize))
        selects = nonzero(root)
        for oldrow, oldcol in zip(selects[0], selects[1]):
            for newdiag in range(scale):
                newrow = oldrow * scale + newdiag
                newcol = oldcol * scale + newdiag
                newroot[newrow, newcol] = root[oldrow, oldcol]
        return newroot
    def contract(root):
        # reduce root matrix size if possible
        oldsize = root.rows
        if oldsize == 1: return root[0,0]
        diags = root.get_diag_blocks()
        subroot = diags[0]
        if subroot.rows < oldsize and not [subroot != diag for diag in diags[1:]].count(True):
            root = type(root)(subroot)
            oldsize = subroot.rows
        if oldsize == 1: return root[0,0]
        if root[oldsize-1, 0] or root[0, oldsize-1]: return root
        factors = []
        for diag in range(2-oldsize, oldsize-1):
            diagrow = - (diag < 0) * diag
            diagcol = (diag > 0) * diag
            value = root[diagrow, diagcol]
            cells = oldsize - (diagrow + diagcol)
            for cell in range(1, cells):
                if root[cell+diagrow, cell+diagcol] != value:
                    return root
            if value:
                factors.append(cells)
        scale = gcd_list(factors)
        if scale < 2: return root
        newsize = oldsize / scale
        if newsize == 1: return root[0,0]
        if newsize < 2: return root
        return type(root)(newsize, newsize, lambda i, j: root[i*scale, j*scale])
    def nzselect(root, nzmap):
        # return matrix of root entries corresponding to nonzero elements of nzmap
        nzmap = Matrix([nzmap])
        order = root.rows
        return Matrix(order, order, lambda i, j: root[i,j] * (nzmap[i,j] and 1 or 0))
    def remain(root):
        # if there are multiple distinct main diagonal values, enlarge and rotate
        order = root.rows
        if order == 1:
            return (type(root)(zeros(2)), Rational(root[0]))
        mdiag = root.nzselect(eye(order))
        if mdiag.is_zero:
            return (root, Rational(0))
        isident = mdiag/mdiag[0]
        for each in range(order):
            isident[each, each] = Rational(isident[each, each])
        if isident.is_Identity:
            newroot = type(root)(Matrix(root) - mdiag)
            return (newroot, Rational(mdiag[0]))
        newroot = type(root)(Matrix(root) - mdiag).enlarge(order)
        for op in range(order):
            np = op * order
            newroot[np:np+order, np:np+order] = mdiag * ell(order)
        return (newroot, Rational(0))
    def scalar_projector(root, initial_depth=16, stepwise=False):
        root, remainder = root.remain()
        projection_multiplier = gcd_list(list(root)) or Rational(1)
        root /= projection_multiplier
        while not root: yield remainder
        order = root.rows
        ration = type(root)(Matrix(root) + eye(order))
        select = list(nonzero(ration.row(0))[1])
        multis = list(ration.extract([0], select))
        step = ration ** order
        while type(step) is not type(root) or step[0] == 1:
            yield remainder
        while step[0] < 1: yield None
        ration = step ** initial_depth
        while True:
            ratios = list(ration.extract([0], select))
            projection = 0
            dividend = Rational(ratios[0])  # pi is off the menu
            for multiplier, divisor in zip(multis[1:], ratios[1:]):
                if multiplier < 0: divisor = -divisor
                projection += multiplier * dividend / divisor
            yield projection * projection_multiplier + remainder
            if not stepwise: step = ration
            ration *= step
    def linear_projector(root, initial_depth=16, stepwise=False, symbolic=True):
        scalar = root.scalar_projector(initial_depth, stepwise)
        first = next(scalar)
        if first is not None:
            yield first
            while True: yield next(scalar)
        mxrat = Matrix(root)
        order = mxrat.rows
        imags = list()
        xjays = symbolic and [1] or [array(eye(order))]
        J = Symbol('J')
        for power in range(1, order):
            selection = root.nzselect(Matrix(ell(order))**power)
            sorter = list(selection)
            sorter.sort()
            if sorter[0] < 0 and sorter[-1] > 0:
                mxrat -= selection
                ximag = selection * Matrix(jay(order))**(-power)
                ximag = Root(ximag).scalar_projector(initial_depth, stepwise)
                imags.append(ximag)
                if symbolic: xjays.append((order > 2) and J(order) or J)
                else: xjays.append(array(jay(order)**power))
        scalar = Root(mxrat).scalar_projector(initial_depth, stepwise)
        for projections in zip(scalar, *imags):
            projection = sum([r*j for r,j in zip(list(projections), xjays)])
            if type(projection) not in (array, ndarray): yield projection
            else: yield type(root)(npmatrix(projection))
    def complex_projector(root, initial_depth=16, stepwise=False):
        from sympy import I
        J = Symbol('J')
        jsubre, jnumre = compile('J\(\d+\)'), compile('\d+')
        depth_factor = 1
        for linear in root.linear_projector(initial_depth, stepwise):
            for jsub in set(jsubre.findall(str(linear))):
                jnum = int(*jnumre.findall(jsub))
                sin,cos = sincos(pi/jnum, initial_depth*depth_factor)
                linear = linear.subs(jsub, cos + J*sin)
            yield linear.subs(J, I)
            depth_factor += stepwise and 1 or depth_factor
    def exponential_projector(root, initial_depth=16):
        order = root.rows
        power = 1
        factorial = 1
        exp = eye(order)
        for i in count(1):
            power *= Matrix(root)
            factorial *= i
            exp += power / factorial
            if i >= initial_depth:
                yield len(exp) > 1 and type(root)(exp) or exp[0]
    def logarithmic_projector(root, initial_depth=16):
        x = root.to_nfloat()
        while not x or x != abs(x): yield None
        twos = 0
        trunc = int(root.to_nfloat())
        while trunc > 2:
            twos += 1
            trunc >>= 1
        if twos:
            root /= Rational(2) ** twos
            lntwo = Root([2]).logarithmic_projector(initial_depth)
        step = 1 - 1/root
        if step < 0:
            lnmin = (~root).logarithmic_projector(initial_depth)
            while True: yield -(next(lnmin)+(twos and twos * next(lntwo)))
        ln = 0
        multiplier = step
        for i in count(1):
            ln += step / Rational(i)
            step *= multiplier
            if i >= initial_depth: yield ln + (twos and twos * next(lntwo))
    def sumseries_projector(root, func, initial_depth=16):
        sum = 0
        for n in count(1):
            sum += func(root, n)
            if n >= initial_depth: yield sum
    def refined_projection(root, projector, precision=16, func=None, as_nfloat=False):
        last = None
        for refinement in func and projector(func) or projector():
            if refinement is None: return None
            if type(refinement) is type(root):
                nfloat_result = refinement.to_nfloat(precision)
            else:
                nfloat_result = nfloat(refinement, precision)
            if nfloat_result == last:
                return as_nfloat and nfloat_result or refinement
            last = nfloat_result
    def exponential_projection(root, precision=16):
        return root.refined_projection(root.exponential_projector, precision)
    def logarithmic_projection(root, precision=16):
        return root.refined_projection(root.logarithmic_projector, precision)
    def sumseries_projection(root, func, precision=16):
        return root.refined_projection(root.sumseries_projector, precision, func)
    def to_bilinear(root):
        order = root.rows
        J,L = [Symbol(x) for x in 'JL']
        if order > 2: J,L = J(order), L(order)
        J,L = [Symbol(str(x), commutative=False) for x in (J,L)]
        total = 0
        diags = []
        for power in range(1,order):
            jpow,lpow = Matrix(jay(order))**power, Matrix(ell(order))**power
            lower = (lpow-jpow)/Rational(2)
            upper = lpow-lower
            diags.append((upper,lower))
        productset = set()
        # This looping construct is not as efficient or ideal as it should be
        for products in product(chain(*((L**power, J**power) for power in range(order))), repeat=order):
            products = [p for p in products if p != 1] or [1]
            productstr = str(products)
            if productstr in productset: continue
            productset.add(productstr)
            factors = [str(element)[0] for element in products]
            nextproducts = False
            for i in range(len(factors)-1):
                if factors[i] == factors[i+1]:
                    nextproducts = True
                    break
            if nextproducts: continue
            xsym = 1
            for x in products: xsym *= x
            xpow = Matrix(ell(order).scale_with(Root.from_bilinear(xsym))[1])
            trial = Root(Matrix(root)/xpow)
            for shift,(upper,lower) in enumerate(diags,1):
                b,c = trial.nzselect(upper), trial.nzselect(lower)
                x = [sum(b.row(row)) for row in range(b.rows-1) if sum(upper.row(row))]
                y = [sum(c.row(row)) for row in range(1,b.rows) if sum(lower.row(row))]
                if (x[0] or y[0]) and x.count(x[0]) == len(x) and y.count(y[0]) == len(y):
                    root = Root(Matrix(root)-(b+c)*xpow)
                    b,c = x[0],y[0]
                    total += J**shift*xsym*(b-c)/Rational(2) + L**shift*xsym*(b+c)/Rational(2)
                    if not root: return total.subs({J**order:-1,L**order:1})
        if root: raise ArithmeticError("Failed to find bilinear expression.")
        return total
    @staticmethod
    def from_bilinear(expr):
        from sympy.core import Add
        from sympy import Function, lambdify
        from sympy.abc import x
        def matjay(x): return Matrix(jay(x))
        def matell(x): return Matrix(ell(x))
        expr = str(expr).replace('J','J(2)').replace('L','L(2)').replace('(2)(','(')
        test = nsimplify(expr)
        if type(test) is Add and test.args[0].is_Number:
            r = 'R'+str(test.atoms(Function).pop())[1:]
            for split in '+-':
                if split not in expr: continue
                elems = []
                for elem in expr.split(split):
                    if len(elem.strip()) and nsimplify(elem).is_Number:
                        elems.append(elem + '*' + r)
                    else: elems.append(elem)
                expr = split.join(elems)
        return Root([lambdify(x, expr, {"R":eye,"J":matjay,"L":matell})(expr)])
    def to_multiset(root):
        from collections import Counter
        from sympy import SparseMatrix
        multiset = Counter()
        for element in SparseMatrix(root).row_list():
            row,col,value = element
            multiset[(row+1,col+1)] = value
        return multiset.elements()
    def to_complex(root, digits=16):
        return root.refined_projection(root.complex_projector, precision=digits, as_nfloat=False)
    def to_nfloat(root, digits=16):
        return root.refined_projection(root.complex_projector, precision=digits, as_nfloat=True)
    def to_Decimal(root):
        nfloat_projection = root.to_nfloat(decimalcontext().prec)
        try: return Decimal(str(nfloat_projection))
        except: return None
    def scale_with(root1, root2):
        root2 = type(root1)([pow(root2,1)])
        r1,r2 = type(root1)([root1.contract()]), type(root1)([root2.contract()])
        if r1.rows == r2.rows: return r1,r2
        newsize = Rational(ilcm(r1.rows, r2.rows))
        s1,s2 = newsize/r1.rows, newsize/r2.rows
        return r1.enlarge(s1), r2.enlarge(s2)
    def contract_sum(root, *args):
        rootsum = root.contract()
        for value in args:
            oldroot = type(root)([rootsum])
            root1, root2 = oldroot.scale_with(value)
            rootsum = type(root)(Matrix(root1) + Matrix(root2)).contract()
        return rootsum
    def distance_add(root, value):
        # Does not yet properly calculate distance for mixed size roots of different values
        root1,root2 = root.scale_with(value)
        order = root1.rows
        nonlinear = False
        testresult = False
        for power in range(order):
            mask = Matrix(ell(order)) ** power
            m1,m2 = root1.nzselect(mask), root2.nzselect(mask)
            if m1.is_zero and m2.is_zero: continue
            if m1.is_zero or m2.is_zero:
                testresult = True
                continue
            ratio = None
            for pos in range(len(m1)):
                if m1[pos] == m2[pos] == 0: continue
                if not m1[pos]:
                    nonlinear = True
                    break
                if ratio is None:
                    ratio = m2[pos]/m1[pos]
                elif ratio != m2[pos]/m1[pos]: 
                    nonlinear = True
                    break
            if nonlinear: break
        if not nonlinear:
            result = root1.contract_sum(root2)
            if testresult and result.to_nfloat() != root1.to_nfloat() + root2.to_nfloat():
                raise ArithmeticError("Failed to calculate distance.")
            return result
        sumexpon = type(root)(Matrix(root1)**order + Matrix(root2)**order)
        for rootprod in range(1, order):
            multiple = binomial(order, rootprod) * ell(order) ** rootprod
            sumexpon = sumexpon.contract_sum(multiple * root1 ** (order - rootprod) * root2 ** rootprod)
        result = sumexpon ** (1/order)
        if testresult and result.to_nfloat() != root1.to_nfloat() + root2.to_nfloat():
            raise ArithmeticError("Failed to calculate distance.")
        return result
    def remain_product(root, value):
        # non-commutative and non-associative but semi-sane for multi roots
        root, remainder = type(root)(root@value).remain()
        return root.contract_sum(remainder)
    @staticmethod
    def settings(**kwargs):
        if kwargs.get('save'):
            save = {'add':Root.__add__, 'mul':Root.__mul__}
            if not hasattr(Root,'_saved'): Root._saved = []
            Root._saved.append(save)
            return
        if kwargs.get('restore'):
            if not hasattr(Root,'_saved'): raise AttributeError("No prior saved state to restore")
            restore = Root._saved.pop()
            if not Root._saved: del Root._saved
            Root.__add__ = restore['add']
            Root.__mul__ = restore['mul']
            return
        if kwargs.keys()-('add','mul'): raise AttributeError('arg!=add|mul')
        if 'add' in kwargs:
            val = kwargs['add']
            if   val == 'contract': Root.__add__ = Root.contract_sum
            elif val == 'distance': Root.__add__ = Root.distance_add
            else: raise ValueError("add!='contract'|'distance'")
        if 'mul' in kwargs:
            val = kwargs['mul']
            if   val == 'matrix': Root.__mul__ = Root.__matmul__
            elif val == 'remain': Root.__mul__ = Root.remain_product
            else: raise ValueError("mul!='matrix'|'remain'")
    def __lt__(root1, root2):
        return float(root1.to_nfloat()) < float(type(root1)([pow(root2,1)]).to_nfloat())
    def __gt__(root1, root2):
        return root1.to_nfloat() > type(root1)([pow(root2,1)]).to_nfloat()
    def __eq__(root1, root2):
        return Matrix([root1.contract()]) == Matrix([type(root1)([pow(root2,1)]).contract()])
    def __le__(root1, root2):
        return root1 == root2 or root1 < root2
    def __ge__(root1, root2):
        return root1 == root2 or root1 > root2
    def __add__(root, value):
        return root.contract_sum(value)
    def __radd__(root, value):
        return root + value
    def __and__(root1, root2):
        return root1.distance_add(root2)
    def __sub__(root1, root2):
        return root1 + -root2
    def __rsub__(root, scalar):
        return -root + scalar
    def __matmul__(root, value):
        root1, root2 = root.scale_with(value)
        return type(root)(Matrix(root1) * Matrix(root2)).contract()
    def __mul__(root, value):
        return root @ value
    def __rmul__(root, scalar):
        return root * scalar
    def __truediv__(root1, root2):
        if isinstance(root2, Root):
            root2 = ~root2
            if root2 is None: return None
            return root1 * root2
        else:
            return root1 * pow(root2, -1)
    def __rtruediv__(root, scalar):
        root = ~root
        if root is None: return None
        return root * scalar
    def __invert__(root):
        try: return type(root)(root.inv())
        except: return None
    def __pow__(root, power):
        return pow(root, power)
    def __neg__(root):
        return root * -1
    def __pos__(root):
        return root
    def __bool__(root):
        return not root.is_zero

class Uniplex(object):
    def __init__(ux, *args):
        if not hasattr(ux, 'units'): raise TypeError("Base helper class does not initialize units")
        if len(args) == 0:
            zeros = [0] * len(ux.units)
            ux.matrix = type(ux)(*zeros).matrix
        elif len(args) == 1:
            arg = args[0]
            if type(arg) is type(ux):
                ux.matrix = arg.matrix
            elif isinstance(arg, Uniplex):
                ux.matrix = type(ux)(arg.matrix).matrix
            elif getattr(arg, 'is_Matrix', 0):
                template = type(ux)()
                if not getattr(arg[0], 'is_Matrix', 0):
                    arg = template._nestmatrix(arg)
                if Matrix(arg) * 0 != template.matrix:
                    raise ValueError("Mismatched matrix dimension")
                ux.matrix = Matrix(arg)
            elif type(arg) in (tuple,list):
                ux.matrix = type(ux)(*arg).matrix
            elif getattr(ux, 'noimag', None) == True:
                args = list(args)+[0]*(len(ux.units)-1)
                ux.matrix = type(ux)(*args).matrix
            else:
                a,aj = _complex_split(*args)
                dftimag = getattr(ux, 'dftimag', None)
                if dftimag:
                    args = [a]+[0]*(len(ux.units)-1)
                    args[dftimag] = aj
                    ux.matrix = type(ux)(*args).matrix
                else:
                    ux.matrix = type(ux)(a,aj).matrix
    def __str__(ux):
        return str(ux.matrix)
    def __repr__(ux):
        return ux.describe() + ': value = ' + ux.value()
    def describe(ux):
        if ux.is_mixed(): return 'Mixed'
        uxtype = str(type(ux))
        uxtype = uxtype[uxtype.find('.'):][1:-2]
        return '%s(%s)' % (uxtype, str(ux.elements()).strip('[]').replace(' ',''))
    def value(ux):
        if ux.is_mixed(): return str(ux)
        elems = [str(x) for x in ux.elements()]
        for elem in range(1, len(elems)):
            if elems[elem][0] is not '-':
                elems[elem] = '+' + elems[elem]
            if '/' in elems[elem]:
                elems[elem] = elems[elem][0]+'('+elems[elem][1:]+')'
        elems = [elem[0]+elem[1] for elem in zip(elems, ux.units)]
        for elem in range(1, len(elems)):
            if elems[elem][1:-len(ux.units[elem])] is '0':
                elems[elem] = ''
            elif elems[elem][1:-len(ux.units[elem])] is '1':
                elems[elem] = elems[elem].replace('1', '')
        elems = [elem for elem in elems if elem]
        if elems[0] is '0' and len(elems) > 1: del elems[0]
        if elems[0][0] is '+': elems[0] = elems[0][1:]
        return ''.join(elems)
    def is_mixed(ux):
        return ux.matrix != type(ux)(ux.elements()).matrix
    @classmethod
    def unit(cls, value):
        if type(value) is int:
            index = value
        else:
            index = next((i for i, x in enumerate(cls.units) if Symbol(x) is Symbol(value)), None)
            if not index: return None
        return cls([0]*(index)+[1]+[0]*(len(cls.units)-(index+1)))
    @classmethod
    def timestable(cls):
        num = len(cls.units)
        m = zeros(num, num)
        for row in range(num):
            for col in range(num):
                r, c = cls.units[row], cls.units[col]
                r = bool(r) and cls.unit(r) or cls(1)
                c = bool(c) and cls.unit(c) or cls(1)
                m[row,col] = Symbol((r*c).value())
        return m
    def flatmatrix(ux):
        def denest(m):
            if getattr(m[0][0], 'is_Matrix', 0):
                m = Matrix([[denest(m[row,col]) for col in range(m.cols)] for row in range(m.rows)])
            return Matrix([[m[orow,ocol][irow,icol] for ocol in range(m.cols) for icol in range(m[0].cols)]
                           for orow in range(m.rows) for irow in range(m[0].rows)])
        m = ux.matrix
        if not getattr(m[0], 'is_Matrix', 0): return m
        return denest(m)
    def _nestmatrix(ux,m):
        def renest(m, template):
            if getattr(template[0], 'is_Matrix', 0): m = renest(m, template[0])
            return Matrix([[m[row:row+template.rows, col:col+template.cols]
                            for col in range(0, m.cols, template.cols)]
                           for row in range(0, m.rows, template.rows)])
        if not getattr(ux.matrix[0], 'is_Matrix', 0): return m
        return renest(m, ux.matrix[0])
    def elements(ux):
        return [ux.matrix[x] for x in range(2)]
    def conjugate(ux):
        elems = ux.elements()
        elems = [elems[0]] + [-elem for elem in elems[1:]]
        return type(ux)(*elems)
    def normalized(ux):
        if hasattr(ux, 'norm'): return ux / ux.norm()
        else: return ux / abs(ux)
    def __abs__(ux):
        return ux.modulus() ** Rational('1/2')
    def __invert__(ux):
        modulus = ux.modulus()
        if not modulus: return None
        if type(modulus) is type(ux):
            remodulus = modulus.modulus()
            if not remodulus: return None
            if type(remodulus) is type(ux):
                plex_modulus = ux.plex_modulus()
                if not plex_modulus: return None
                if type(plex_modulus.modulus()) is type(ux):
                    raise ArithmeticError
                return ux.plex_conjugate() / plex_modulus
        return ux.conjugate() / modulus
    def __add__(ux1, ux2):
        if type(ux2) is not type(ux1) and isinstance(ux2, UniplexDuo): return ux2+ux1
        return type(ux1)(ux1.matrix + type(ux1)(ux2).matrix)
    def __radd__(ux, scalar):
        return ux + scalar
    def __sub__(ux1, ux2):
        return ux1 + (-ux2)
    def __rsub__(ux, scalar):
        return -ux + scalar
    def __mul__(ux1, ux2):
        if isinstance(ux2, Uniplex):
            if isinstance(ux2, UniplexDuo): return ux2.__rmul__(ux1)
            return type(ux1)(ux1.matrix * type(ux1)(ux2).matrix)
        if type(ux2) is float: ux2 = Rational(ux2)
        return type(ux1)(ux1.matrix * ux2)
    def __rmul__(ux, scalar):
        return ux * scalar
    def __truediv__(ux1, ux2):
        if isinstance(ux2, Uniplex):
            ux2 = ~ux2
            if ux2 is None: return None
            return type(ux1)(ux1 * ux2)
        else:
            return type(ux1)(ux1 * ux2 ** -1)
    def __rtruediv__(ux, scalar):
        ux = ~ux
        if ux is None: return None
        return ux * scalar
    def __pow__(ux, p):
        return pow(ux, p)
    def __eq__(ux1, ux2):
        return type(ux1) is type(ux2) and ux1.matrix == ux2.matrix
    def __ne__(ux1, ux2):
        return not ux1 == ux2
    def __neg__(ux):
        return ux * -1
    def __pos__(ux):
        return ux
    def __bool__(ux):
        return not ux.matrix.is_zero

class SquareUniplex(Uniplex):
    def __init__(squx, *args):
        super().__init__(*args)
        if hasattr(squx, 'matrix'): return
        if len(args) == 3 and len(squx.units) > 3:
            a,b,c = args
            squx.matrix = type(squx)(0,a,b,c).matrix

class CubicUniplex(Uniplex):
    noimag = True
    def __init__(cuux, *args):
        super().__init__(*args)
        if hasattr(cuux, 'matrix'): return
        if len(args) == 2:
            a,b = args
            cuux.matrix = type(cuux)(a,b,0).matrix
    def __invert__(cux):
        if not cux.matrix.det(): return None
        return type(cux)(cux.matrix.inv())

class Duplex(Uniplex):
    def elements(biux):
        elems = [x.elements() for x in biux.plex_elements()]
        return [elem for sublist in elems for elem in sublist]
    def plex_elements(biux):
        return [biux.subtype(Matrix(biux.matrix[x])) for x in range(2)]
    def plex_conjugate(biux):
        elems = [x.conjugate().elements() for x in biux.plex_elements()]
        elems = [elem for sublist in elems for elem in sublist]
        return type(biux)(elems)
    def modulus(biux):
        biuxsum = biux * biux.conjugate()
        if not biuxsum: return None
        elems = biuxsum.elements()
        for elem in elems[1:]:
            if elem: return biuxsum
        return elems[0]
    def plex_modulus(biux):
        return biux * biux.plex_conjugate()
class SquareDuplex(Duplex): pass

class UniplexDuo(Duplex):
    def conjugate(uxduo):
        elems = uxduo.elements()
        return type(uxduo)([elems[0]]+[-elem for elem in elems[1:]])
    def __add__(uxduo1, uxduo2):
        if type(uxduo1) is type(uxduo2) or not isinstance(uxduo2, Uniplex):
            return super().__add__(uxduo2)
        m1, m2 = uxduo1.flatmatrix(), uxduo2.flatmatrix()
        if m1.rows != m2.rows: raise ValueError("Mismatched matrix dimension")
        if len(m1) < len(m2): uxduo1,m1,uxduo2,m2 = uxduo2,m2,uxduo1,m1
        for row in range(m2.rows):
            for col in range(m2.cols):
                m1[row,col] += m2[row,col]
        return type(uxduo1)(m1)
    def __mul__(uxduo1, uxduo2):
        if type(uxduo2) is not type(uxduo1):
            if isinstance(uxduo2, UniplexDuo):
                if len(uxduo1.flatmatrix()) < len(uxduo2.flatmatrix()): uxduo1,uxduo2 = uxduo2,uxduo1
            return type(uxduo1)(Matrix([[(sub * uxduo2).matrix for sub in uxduo1.plex_elements()]]))
        a,b,c,d = uxduo1.plex_elements()+uxduo2.plex_elements()
        cx_mul = getattr(uxduo1, 'cx_mul', False)
        if not cx_mul: return type(uxduo1)(Matrix([[(a*c+d*b).matrix, (d*a+b*c).matrix]]))
        return type(uxduo1)(Matrix([[(a*c-d.conjugate()*b).matrix, (d*a+b*c.conjugate()).matrix]]))
    def __rmul__(uxduo, ux):
        return type(uxduo)(Matrix([[(ux * sub).matrix for sub in uxduo.plex_elements()]]))

class SquareUniplexDuo(UniplexDuo):
    def __init__(uxduo, *args):
        super().__init__(*args)
        if hasattr(uxduo, 'matrix'): return
        if len(args) == 2:
            r,j = args
            uxduo.matrix = type(uxduo)(r,0,j,0).matrix
        elif len(args) == 3:
            i,j,k = args
            uxduo.matrix = type(uxduo)(0,i,j,k).matrix

class CubicUniplexDuo(UniplexDuo):
    def __init__(cuuxduo, *args):
        super().__init__(*args)
        if hasattr(cuuxduo, 'matrix'): return
        if len(args) == 2:
            r,j = args
            cuuxduo.matrix = type(cuuxduo)(r,0,0,j,0,0).matrix

class CubicDuplex(Duplex):
    noimag = True
    def __invert__(bicux):
        return type(bicux)(bicux.flatmatrix().inv())

class Quadplex(Duplex):
    def plex_elements(quadux):
        return [quadux.subtype(Matrix(quadux.matrix[x][y])) for x in range(2) for y in range(2)]
    def conjugate(quadux):
        a,b,c,d = quadux.plex_elements()
        elems = [x.elements() for x in [a,-b,-c,-d]]
        elems = [elem for sublist in elems for elem in sublist]
        return type(quadux)(elems)
class SquareQuadplex(Quadplex): pass

class Perplex(SquareUniplex):
    units = ['','h']
    def __init__(px, *args):
        super().__init__(*args)
        if hasattr(px, 'matrix'): return
        if len(args) == 2:
            a,b = _rationalized(*args)
            px.matrix = Matrix([[a,b], [b,a]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def modulus(px):
        return (px * px.conjugate()).matrix[0]
class SplitComplex(Perplex): pass

class GaussianRational(SquareUniplex):
    units = ['','i']
    def __init__(cx, *args):
        super().__init__(*args)
        if hasattr(cx, 'matrix'): return
        if len(args) == 2:
            a,b = _rationalized(*args)
            if not b: a,b = _complex_split(a)
            cx.matrix = Matrix([[a,b], [-b,a]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def modulus(cx):
        return sum(x ** 2 for x in cx.elements())
    def norm(cx):
        return abs(cx)
class Complex(GaussianRational):
    units = ['','j']

class Dual(SquareUniplex):
    units = ['','ε']
    def __init__(dl, *args):
        super().__init__(*args)
        if hasattr(dl, 'matrix'): return
        if len(args) == 2:
            a,b = _rationalized(*args)
            dl.matrix = Matrix([[a,b], [0,a]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def modulus(dl):
        return dl.matrix[0] ** 2
    def norm(dl):
        return (dl.modulus() + dl.matrix[1] ** 2) ** Rational('1/2')

class CubicPerplex(CubicUniplex):
    units = ['','h','hh']
    noimag = True
    def __init__(cupx, *args):
        super().__init__(*args)
        if hasattr(cupx, 'matrix'): return
        if len(args) == 3:
            r,h,hh = _rationalized(*args)
            cupx.matrix = Matrix([[r,h,hh],[hh,r,h],[h,hh,r]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def elements(cupx):
        m = cupx.matrix
        r  = (m[0]+m[4]+m[8]) / Rational(3)
        l  = (m[1]+m[5]+m[6]) / Rational(3)
        ll = (m[2]+m[3]+m[7]) / Rational(3)
        return [r,l,ll]
class SplitCubicComplex(CubicPerplex): pass

class CubicComplex(CubicUniplex):
    units = ['','j','jj']
    def __init__(cucx, *args):
        super().__init__(*args)
        if hasattr(cucx, 'matrix'): return
        if len(args) == 3:
            r,j,jj = _rationalized(*args)
            cucx.matrix = Matrix([[r,j,jj],[-jj,r,j],[-j,-jj,r]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def elements(cucx):
        m = cucx.matrix
        r  = (m[0]+m[4]+m[8]) / Rational(3)
        j  = (m[1]+m[5]-m[6]) / Rational(3)
        jj = (m[2]-m[3]-m[7]) / Rational(3)
        return [r,j,jj]

class CubicDual(CubicUniplex):
    units = ['','ε','εε']
    def __init__(cudl, *args):
        super().__init__(*args)
        if hasattr(cudl, 'matrix'): return
        if len(args) == 3:
            r,e,ee = _rationalized(*args)
            cudl.matrix = Matrix([[r,e,ee],[ee*0,r,e],[e*0,ee*0,r]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def elements(cudl):
        m = cudl.matrix
        r = (m[0]+m[4]+m[8]) / Rational(3)
        e = (m[1]+m[5]) / Rational(2)
        ee = m[2]
        return [r,e,ee]
class Trial(CubicDual): pass

class Quartal(SquareUniplex):
    units = ['','J','L','JL']
    dftimag = 1
    def __init__(qt, *args):
        super().__init__(*args)
        if hasattr(qt, 'matrix'): return
        if len(args) == 4:
            LL,J,L,JL = _rationalized(*args)
            if not J: LL,J = _complex_split(LL)
            qt.matrix = Matrix([[LL+JL, L+J], [L-J, LL-JL]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def elements(qt):
        a,b,c,d = qt.matrix
        return [elem / Rational(2) for elem in [a+d, b-c, b+c, a-d]]
    def modulus(qt):
        return qt.matrix.det()
    def norm(qt):
        return abs(qt)
class Coquaternion(Quartal): units = ['','i','j','k']
class SplitQuaternion(Coquaternion): pass

class ComplexPerplex(Quartal):
    units = ['','h','j','hj']
    def __init__(cxpx, *args):
        if len(args) == 4:
            r,h,j,hj = args
            args = r,j,h,-hj
        super().__init__(*args)
    def elements(cxpx):
        r,j,h,jh = super().elements()
        return [r,h,j,-jh]
class ComplexSplitComplex(ComplexPerplex): pass

# CubicQuartal not implemented: Unusable timestable, i.e.:
#   ⎡ 1        J          JJ          L          LL         JL         JLJL        JJL        JLL   ⎤
#   ⎢ J        JJ         -1         JL         JLL         JJL      JJ+LL-JL      -L      J-L-JLJL ⎥
#   ⎢ JJ       -1         -J         JJL      J-L-JLJL      -L      -1-JJL+JLL     -JL        -LL   ⎥
#   ⎢ L     JJ+LL-JL     -JLL        LL          1       1+JJL-JLL      JJ         -J        -JLJL  ⎥
#   ⎢ LL      -JJL       JLJL         1          L       -J+L+JLJL     -JLL     -JJ-LL+JL     -JJ   ⎥
#   ⎢ JL   -1-JJL+JLL  -J+L+JLJL     JLL         J         JLJL         -1         -JJ     -JJ-LL+JL⎥
#   ⎢JLJL     -LL         JJL     -JJ-LL+JL  -1-JJL+JLL     -1         -JL      J-L-JLJL      -L    ⎥
#   ⎢JJL     -JLJL        LL      J-L-JLJL       JJ      JJ+LL-JL       -J          1      1+JJL-JLL⎥
#   ⎣JLL       L       JJ+LL-JL       J          JL         LL      -J+L+JLJL   1+JJL-JLL      1    ⎦

class PerplexPerplexDuo(SquareUniplexDuo):
    units = ['','i','j','k']
    subtype = Perplex
    noimag = True
    def __init__(pxpxpx, *args):
        super().__init__(*args)
        if hasattr(pxpxpx, 'matrix'): return
        if len(args) == 4:
            r,i,j,k = _rationalized(*args)
            pxpxpx.matrix = Matrix([[pxpxpx.subtype(r,i).matrix, pxpxpx.subtype(j,k).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
class PerplexDuo(PerplexPerplexDuo): pass
class SplitComplexDuo(PerplexDuo): pass
class Biperplex(PerplexDuo): pass

class ComplexPerplexPerplexDuo(SquareUniplexDuo):
    # Isometric to a modified Quartal with L before J and LJ instead of JL
    # Implemented anyhow for illustrative purposes
    units = ['','h','j','hj']
    subtype = Perplex
    cx_mul = True
    def __init__(cxpxpx, *args):
        super().__init__(*args)
        if hasattr(cxpxpx, 'matrix'): return
        if len(args) == 4:
            r,h,j,hj = _rationalized(*args)
            cxpxpx.matrix = Matrix([[cxpxpx.subtype(r,h).matrix, cxpxpx.subtype(j,hj).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
class ComplexPerplexDuo(ComplexPerplexPerplexDuo): pass
class ComplexSplitComplexDuo(ComplexPerplexDuo): pass
# ComplexPerplex/ComplexSplitComplex implemented as modified Quartal for efficiency

class PerplexComplexDuo(SquareUniplexDuo):
    units = ['','h','j','hj']
    subtype = Complex
    def __init__(pxcxcx, *args):
        super().__init__(*args)
        if hasattr(pxcxcx, 'matrix'): return
        if len(args) == 4:
            r,h,j,hj = _rationalized(*args)
            pxcxcx.matrix = Matrix([[pxcxcx.subtype(r,j).matrix, pxcxcx.subtype(h,hj).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def elements(pxcxcx):
        elems = [x.elements() for x in pxcxcx.plex_elements()]
        return [elem[0] for elem in elems] + [elem[1] for elem in elems]
    def plex_conjugate(pxcxcx):
        elems = [x.conjugate().elements() for x in pxcxcx.plex_elements()]
        elems = [elem[0] for elem in elems] + [elem[1] for elem in elems]
        return type(pxcxcx)(elems)
class Dycomplex(PerplexComplexDuo): pass
class SplitBicomplex(Dycomplex): pass
class PerplexComplex(Dycomplex): pass

class ComplexDuo(SquareUniplexDuo):
    units = ['','i','j','k']
    subtype = Complex
    cx_mul = True
    def __init__(cxcxcx, *args):
        super().__init__(*args)
        if hasattr(cxcxcx, 'matrix'): return
        if len(args) == 4:
            r,i,j,k = _rationalized(*args)
            cxcxcx.matrix = Matrix([[cxcxcx.subtype(r,i).matrix, cxcxcx.subtype(j,k).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def norm(cxcxcx):
        return abs(cxcxcx)
class Quaternion(ComplexDuo): pass
class ComplexComplexDuo(ComplexDuo): pass

class PerplexDualDualDuo(SquareUniplexDuo):
    units = ['','h','ε','εh']
    subtype = Dual
    noimag = True
    def __init__(pxdldl, *args):
        super().__init__(*args)
        if hasattr(pxdldl, 'matrix'): return
        if len(args) == 4:
            r,h,e,eh = _rationalized(*args)
            pxdldl.matrix = Matrix([[pxdldl.subtype(r,e).matrix, pxdldl.subtype(h,eh).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def elements(pxdldl):
        elems = [x.elements() for x in pxdldl.plex_elements()]
        return [elem[0] for elem in elems] + [elem[1] for elem in elems]
class PerplexDualDuo(PerplexDualDualDuo): pass
class DualDuo(PerplexDualDuo): pass

class ComplexDualDualDuo(SquareUniplexDuo):
    units = ['','j','ε','εj']
    subtype = Dual
    dftimag = 1
    cx_mul = True
    def __init__(cxdldl, *args):
        super().__init__(*args)
        if hasattr(cxdldl, 'matrix'): return
        if len(args) == 4:
            r,j,e,ej = _rationalized(*args)
            cxdldl.matrix = Matrix([[cxdldl.subtype(r,e).matrix, cxdldl.subtype(j,ej).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def elements(cxdldl):
        elems = [x.elements() for x in cxdldl.plex_elements()]
        return [elem[0] for elem in elems] + [elem[1] for elem in elems]
class ComplexDualDuo(ComplexDualDualDuo): pass

class PerplexQuartalQuartalDuo(SquareUniplexDuo):
    units = ['','J','L','JL','h','hJ','hL','hJL']
    subtype = Quartal
    dftimag = 1
    def __init__(pxqtqt, *args):
        super().__init__(*args)
        if hasattr(pxqtqt, 'matrix'): return
        if len(args) == 4:
            r,J,L,JL = args
            pxqtqt.matrix = type(pxqtqt)(r,J,L,JL,0,0,0,0).matrix
        elif len(args) == 8:
            r,J,L,JL,h,hJ,hL,hJL = _rationalized(*args)
            pxqtqt.matrix = Matrix([[pxqtqt.subtype(r,J,L,JL).matrix, pxqtqt.subtype(h,hJ,hL,hJL).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    # inverting this is still a challenge to be figured out
class PerplexQuartalDuo(PerplexQuartalQuartalDuo): pass

class ComplexQuartalQuartalDuo(SquareUniplexDuo):
    units = ['','J','L','JL','h','hJ','hL','hJL']
    subtype = Quartal
    dftimag = 1
    cx_mul = True
    def __init__(cxqtqt, *args):
        super().__init__(*args)
        if hasattr(cxqtqt, 'matrix'): return
        if len(args) == 4:
            r,J,L,JL = args
            cxqtqt.matrix = type(cxqtqt)(r,J,L,JL,0,0,0,0).matrix
        elif len(args) == 8:
            r,J,L,JL,h,hJ,hL,hJL = _rationalized(*args)
            cxqtqt.matrix = Matrix([[cxqtqt.subtype(r,J,L,JL).matrix, cxqtqt.subtype(h,hJ,hL,hJL).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
class ComplexQuartalDuo(ComplexQuartalQuartalDuo): pass

class ComplexComplexPerplexDuo(ComplexQuartalDuo):
    units = ['','i','j','k','ℓ','ℓi','ℓj','ℓk']
    def __init__(cxcxpx, *args):
        if len(args) == 8:
            r,i,j,k,l,li,lj,lk = args
            args = r,j,l,-lj,i,-k,li,-lk
        super().__init__(*args)
    def elements(cxcxpx):
        r,j,l,jl,i,nk,li,kl = super().elements()
        return r,i,j,-nk,l,li,-jl,-kl
class ComplexComplexSplitComplexDuo(ComplexComplexPerplexDuo): pass
class SplitOctonion(ComplexComplexPerplexDuo): pass

class PerplexPerplexPerplexDuoDuo(SquareUniplexDuo):
    units = ['','i','j','ij','k','ik','jk','ijk']
    subtype = PerplexDuo
    noimag = True
    def __init__(pxpxduo, *args):
        super().__init__(*args)
        if hasattr(pxpxduo, 'matrix'): return
        if len(args) == 4:
            r,i,j,k = args
            pxpxduo.matrix = type(pxpxduo)(r,i,j,0,k,0,0,0).matrix
        elif len(args) == 8:
            r,i,j,ij,k,ik,jk,ijk = _rationalized(*args)
            pxpxduo.matrix = Matrix([[pxpxduo.subtype(r,i,j,ij).matrix, pxpxduo.subtype(k,ik,jk,ijk).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def __invert__(pxpxduo):
        a,b = pxpxduo.plex_elements()
        idet = ~(a*a-b*b)
        if not idet: return None
        return type(pxpxduo)(Matrix([[(idet*a).matrix, (-idet*b).matrix]]))
class PerplexPerplexDuoDuo(PerplexPerplexPerplexDuoDuo): pass
class PerplexDuoDuo(PerplexPerplexDuoDuo): pass

class ComplexComplexComplexDuoDuo(SquareUniplexDuo):
    units = ['','i','j','ij','k','ik','jk','ijk']
    subtype = ComplexDuo
    dftimag = 1
    cx_mul = True
    def __init__(cxcxduo, *args):
        super().__init__(*args)
        if hasattr(cxcxduo, 'matrix'): return
        if len(args) == 4:
            r,i,j,k = args
            cxcxduo.matrix = type(cxcxduo)(r,i,j,0,k,0,0,0).matrix
        elif len(args) == 8:
            r,i,j,ij,k,ik,jk,ijk = _rationalized(*args)
            cxcxduo.matrix = Matrix([[cxcxduo.subtype(r,i,j,ij).matrix, cxcxduo.subtype(k,ik,jk,ijk).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
class ComplexComplexDuoDuo(ComplexComplexComplexDuoDuo): pass
class ComplexDuoDuo(ComplexComplexDuoDuo): pass
class Octonion(ComplexDuoDuo): pass

class PerplexCubicPerplexCubicPerplexDuo(CubicUniplexDuo):
    units = ['','h','hh','j','jh','jhh']
    subtype = CubicPerplex
    noimag = True
    def __init__(pxcupxcupx, *args):
        super().__init__(*args)
        if hasattr(pxcupxcupx, 'matrix'): return
        if len(args) == 6:
            r,h,hh,j,jh,jhh = _rationalized(*args)
            pxcupxcupx.matrix = Matrix([[pxcupxcupx.subtype(r,h,hh).matrix, pxcupxcupx.subtype(j,jh,jhh).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def __invert__(pxcupxcupx):
        a,b = pxcupxcupx.plex_elements()
        idet = ~(a*a-b*b)
        if not idet: return None
        return type(pxcupxcupx)(Matrix([[(idet*a).matrix, (-idet*b).matrix]]))
class PerplexCubicPerplexDuo(PerplexCubicPerplexCubicPerplexDuo): pass

class ComplexCubicPerplexCubicPerplexDuo(CubicUniplexDuo):
    units = ['','h','hh','j','hj','hhj']
    subtype = CubicPerplex
    hasimag = 3
    cx_mul = True
    def __init__(cxcupxcupx, *args):
        super().__init__(*args)
        if hasattr(cxcupxcupx, 'matrix'): return
        if len(args) == 6:
            r,h,hh,j,hj,hhj = _rationalized(*args)
            cxcupxcupx.matrix = Matrix([[cxcupxcupx.subtype(r,h,hh).matrix, cxcupxcupx.subtype(j,hj,hhj).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    # inverting this is still a challenge to be figured out
class ComplexCubicPerplexDuo(ComplexCubicPerplexCubicPerplexDuo): pass

class PerplexCubicComplexCubicComplexDuo(CubicUniplexDuo):
    units = ['','j','jj','h','jh','jjh']
    subtype = CubicComplex
    noimag = True
    def __init__(pxcucxcucx, *args):
        super().__init__(*args)
        if hasattr(pxcucxcucx, 'matrix'): return
        if len(args) == 6:
            r,j,jj,h,jh,jjh = _rationalized(*args)
            pxcucxcucx.matrix = Matrix([[pxcucxcucx.subtype(r,j,jj).matrix, pxcucxcucx.subtype(h,jh,jjh).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def __invert__(pxcucxcucx):
        a,b = pxcucxcucx.plex_elements()
        idet = ~(a*a-b*b)
        if not idet: return None
        return type(pxcucxcucx)(Matrix([[(idet*a).matrix, (-idet*b).matrix]]))
class PerplexCubicComplexDuo(PerplexCubicComplexCubicComplexDuo): pass

class ComplexCubicComplexCubicComplexDuo(CubicUniplexDuo):
    units = ['','h','hh','j','hj','hhj']
    subtype = CubicComplex
    hasimag = 3
    cx_mul = True
    def __init__(cxcucxcucx, *args):
        super().__init__(*args)
        if hasattr(cxcucxcucx, 'matrix'): return
        if len(args) == 6:
            r,h,hh,j,hj,hhj = _rationalized(*args)
            cxcucxcucx.matrix = Matrix([[cxcucxcucx.subtype(r,h,hh).matrix, cxcucxcucx.subtype(j,hj,hhj).matrix]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    # inverting this is still a challenge to be figured out
class ComplexCubicComplexDuo(ComplexCubicComplexCubicComplexDuo): pass

# Biperplex behaves identically to PerplexDuo, so does not need separate implementation
# Dycomplex behaves identically to PerplexComplexDuo, so does not need separate implementation

class Bicomplex(SquareDuplex):
    units = ['','h','j','hj']
    subtype = Complex
    def __init__(cxcx, *args):
        super().__init__(*args)
        if hasattr(cxcx, 'matrix'): return
        if len(args) == 2:
            a,aj,b,bj = _complex_split(*args)
            cxcx.matrix = type(cxcx)(a,aj,b,bj).matrix
        elif len(args) == 4:
            r,h,j,hj = _rationalized(*args)
            a,b = cxcx.subtype(r,h), cxcx.subtype(j,hj)
            cxcx.matrix = Matrix([[a,b], [-b,a]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))

class Tessarine(Bicomplex):
    units = ['','i','j','k']
    def __init__(cxcx, *args):
        if len(args) == 4:
            r,i,j,k = args
            args = r,i,-k,j
        super().__init__(*args)
    def elements(cxcx):
        r,i,nk,j = super().elements()
        return r,i,j,-nk

class DualComplex(SquareDuplex):
    units = ['','j','ε','εj']
    subtype = Dual
    dftimag = 1
    def __init__(dlcx, *args):
        super().__init__(*args)
        if hasattr(dlcx, 'matrix'): return
        if len(args) == 2:
            a,b = args
            dlcx.matrix = type(dlcx)(a,b,0,0).matrix
        elif len(args) == 4:
            r,j,e,ej = _rationalized(*args)
            a,b = dlcx.subtype(r,e), dlcx.subtype(j,ej)
            dlcx.matrix = Matrix([[a,b], [-b,a]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def elements(dlcx):
        elems = [x.elements() for x in dlcx.plex_elements()]
        return [elem[0] for elem in elems] + [elem[1] for elem in elems]
    def conjugate(dlcx):
        a,b = dlcx.plex_elements()
        elems = [x.elements() for x in [a,-b]]
        elems = [elem[0] for elem in elems] + [elem[1] for elem in elems]
        return type(dlcx)(elems)
    def plex_conjugate(dlcx):
        elems = [x.conjugate().elements() for x in dlcx.plex_elements()]
        elems = [elem[0] for elem in elems] + [elem[1] for elem in elems]
        return type(dlcx)(elems)

class QuartalComplex(SquareDuplex):
    units = ['','h','J','hJ','L','hL','JL','hJL']
    subtype = Quartal
    dftimag = 2
    def __init__(qtcx, *args):
        super().__init__(*args)
        if hasattr(qtcx, 'matrix'): return
        if len(args) == 2:
            a,b = args
            qtcx.matrix = type(qtcx)(a,b,0,0).matrix
        elif len(args) == 4:
            a,aj,b,bj,c,cj,d,dj = _complex_split(*args)
            qtcx.matrix = type(qtcx)(a,aj,b,bj,c,cj,d,dj).matrix
        elif len(args) == 8:
            r,h,J,hJ,L,hL,JL,hJL = _rationalized(*args)
            a,b = qtcx.subtype(r,J,L,JL), qtcx.subtype(h,hJ,hL,hJL)
            qtcx.matrix = Matrix([[a,b], [-b,a]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def elements(qtcx):
        elems = [x.elements() for x in qtcx.plex_elements()]
        elems = zip(elems[0], elems[1])
        return [elem for sublist in elems for elem in sublist]
    def conjugate(qtcx):
        a,b = qtcx.plex_elements()
        elems = [x.elements() for x in [a,-b]]
        elems = zip(elems[0], elems[1])
        elems = [elem for sublist in elems for elem in sublist]
        return type(qtcx)(elems)
    def plex_conjugate(qtcx):
        elems = [x.conjugate().elements() for x in qtcx.plex_elements()]
        elems = zip(elems[0], elems[1])
        elems = [elem for sublist in elems for elem in sublist]
        return type(qtcx)(elems)

class Bidual(SquareDuplex):
    units = ['','ε','ε2','εε2']
    subtype = Dual
    noimag = True
    def __init__(dldl, *args):
        super().__init__(*args)
        if hasattr(dldl, 'matrix'): return
        if len(args) == 2:
            a,b = args
            dldl.matrix = type(dldl)(a,b,0,0).matrix
        elif len(args) == 3:
            a,b,c = args
            dldl.matrix = type(dldl)(0,a,b,c).matrix
        elif len(args) == 4:
            r,e,e2,ee2 = _rationalized(*args)
            a,b = dldl.subtype(r,e), dldl.subtype(e2,ee2)
            dldl.matrix = Matrix([[a,b], [b*0,a]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
class Hyperdual(Bidual): pass

class BicubicPerplex(CubicDuplex):
    units = ['','h','hh','j','hj','hhj','jj','hjj','hhjj']
    subtype = CubicPerplex
    def __init__(cupxcupx, *args):
        super().__init__(*args)
        if hasattr(cupxcupx, 'matrix'): return
        if len(args) == 1:
            args = list(args)+[0]*8
            cupxcupx.matrix = type(cupxcupx)(*args).matrix
        if len(args) == 2:
            a,b = args
            cupxcupx.matrix = type(cupxcupx)(a,b,0).matrix
        elif len(args) == 3:
            a,b,c = args
            cupxcupx.matrix = type(cupxcupx)(a,0,0,b,0,0,c,0,0).matrix
        elif len(args) == 9:
            r,h,hh,j,hj,hhj,jj,hjj,hhjj = _rationalized(*args)
            r  = Matrix([[r,h,hh],      [hh,r,h],      [h,hh,r]])
            j  = Matrix([[j,hj,hhj],    [hhj,j,hj],    [hj,hhj,j]])
            jj = Matrix([[jj,hjj,hhjj], [hhjj,jj,hjj], [hjj,hhjj,jj]])
            cupxcupx.matrix = Matrix([[r,j,jj], [jj,r,j], [j,jj,r]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def plex_elements(cupxcupx):
        m = cupxcupx.matrix
        r  = (m[0]+m[4]+m[8]) / Rational(3)
        j  = (m[1]+m[5]+m[6]) / Rational(3)
        jj = (m[2]+m[3]+m[7]) / Rational(3)
        return [cupxcupx.subtype(x) for x in (r,j,jj)]
class SplitBicubicComplex(BicubicPerplex): pass

class BicubicComplex(CubicDuplex):
    units = ['','h','hh','j','hj','hhj','jj','hjj','hhjj']
    subtype = CubicComplex
    def __init__(cucxcucx, *args):
        super().__init__(*args)
        if hasattr(cucxcucx, 'matrix'): return
        if len(args) == 1:
            args = list(args)+[0]*8
            cucxcucx.matrix = type(cucxcucx)(*args).matrix
        if len(args) == 2:
            a,b = args
            cucxcucx.matrix = type(cucxcucx)(a,b,0).matrix
        elif len(args) == 3:
            a,b,c = args
            cucxcucx.matrix = type(cucxcucx)(a,0,0,b,0,0,c,0,0).matrix
        elif len(args) == 9:
            r,h,hh,j,hj,hhj,jj,hjj,hhjj = _rationalized(*args)
            r  = Matrix([[r,h,hh],      [-hh,r,h],      [-h,-hh,r]])
            j  = Matrix([[j,hj,hhj],    [-hhj,j,hj],    [-hj,-hhj,j]])
            jj = Matrix([[jj,hjj,hhjj], [-hhjj,jj,hjj], [-hjj,-hhjj,jj]])
            cucxcucx.matrix = Matrix([[r,j,jj], [-jj,r,j], [-j,-jj,r]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def plex_elements(cucxcucx):
        m = cucxcucx.matrix
        r  = (m[0]+m[4]+m[8]) / Rational(3)
        j  = (m[1]+m[5]-m[6]) / Rational(3)
        jj = (m[2]-m[3]-m[7]) / Rational(3)
        return [cucxcucx.subtype(x) for x in (r,j,jj)]

# BicubicDual not implemented: Lacks closure under multiplication

class PerplexQuartal(SquareDuplex):
    units = ['','h','J','hJ','L','hL','JL','hJL']
    subtype = Perplex
    dftimag = 2
    def __init__(pxqt, *args):
        super().__init__(*args)
        if hasattr(pxqt, 'matrix'): return
        if len(args) == 2:
            a,b = args
            pxqt.matrix = type(pxqt)(a,b,0,0).matrix
        elif len(args) == 3:
            a,b,c = args
            pxqt.matrix = type(pxqt)(0,a,b,c).matrix
        elif len(args) == 4:
            a,b,c,d = args
            pxqt.matrix = type(pxqt)(a,0,b,0,c,0,d,0).matrix
        elif len(args) == 8:
            args = _rationalized(*args)
            r,J,L,JL = [pxqt.subtype(args[x], args[x+1]).matrix for x in range(0,8,2)]
            pxqt.matrix = Matrix([[r+JL, L+J], [L-J, r-JL]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def modulus(qtqt):
        return qtqt.flatmatrix().det()
    def plex_elements(pxqt):
        a,b,c,d = pxqt.matrix
        return [Perplex(elem / Rational(2)) for elem in [a+d, b-c, b+c, a-d]]
    def __invert__(qtqt):
        return type(qtqt)(qtqt.flatmatrix().inv())

# DualQuartal not implemented: any(ε,εJ,εL,εJL)*any(ε,εJ,εL,εJL) = 0

class Biquartal(SquareDuplex):
    units = ['',   'hJ',   'hL',   'hJL',
             'J',  'JhJ',  'JhL',  'JhJL',
             'L',  'LhJ',  'LhL',  'LhJL',
             'JL', 'JLhJ', 'JLhL', 'JLhJL']
    subtype = Quartal
    dftimag = 4
    def __init__(qtqt, *args):
        super().__init__(*args)
        if hasattr(qtqt, 'matrix'): return
        if len(args) == 2:
            a,b = args
            qtqt.matrix = type(qtqt)(a,b,0,0).matrix
        elif len(args) == 3:
            a,b,c = args
            qtqt.matrix = type(qtqt)(0,a,b,c).matrix
        elif len(args) == 4:
            args = [arg for args in [[arg]+[0]*3 for arg in args] for arg in args]
            qtqt.matrix = type(qtqt)(*args).matrix
        elif len(args) == 16:
            r,J,L,JL = [Quartal(*args).matrix for args in [_rationalized(*args)[row:row+4] for row in range(0,16,4)]]
            qtqt.matrix = Matrix([[r+JL, L+J], [L-J, r-JL]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def plex_elements(qtqt):
        a,b,c,d = qtqt.matrix
        return [qtqt.subtype(elem / Rational(2)) for elem in [a+d, b-c, b+c, a-d]]
    def modulus(qtqt):
        return qtqt.flatmatrix().det()
    def __invert__(qtqt):
        return type(qtqt)(qtqt.flatmatrix().inv())
class Bicoquaternion(Biquartal):
    units = ['',  'hi',  'hj',  'hk',
             'i', 'ihi', 'ihj', 'ihk',
             'j', 'jhi', 'jhj', 'jhk',
             'k', 'khi', 'khj', 'khk']

class Dyquaternion(SquareQuadplex):
    units = ['','h','i','hi','j','hj','k','hk']
    subtype = Perplex
    dftimag = 4
    def __init__(pxqn, *args):
        super().__init__(*args)
        if hasattr(pxqn, 'matrix'): return
        if len(args) == 2:
            a,b = args
            pxqn.matrix = type(pxqn)(a,0,b,0).matrix
        elif len(args) == 3:
            a,b,c = args
            pxqn.matrix = type(pxqn)(0,a,b,c).matrix
        elif len(args) == 4:
            a,b,c,d = args
            pxqn.matrix = type(pxqn)(a,0,b,0,c,0,d,0).matrix
        elif len(args) == 8:
            r,h,i,hi,j,hj,k,hk = _rationalized(*args)
            a,b,c,d = pxqn.subtype(r,h), pxqn.subtype(i,hi), pxqn.subtype(j,hj), pxqn.subtype(k,hk)
            pxqn.matrix = Matrix([[Complex(a,b), Complex(c,d)], [Complex(-c,d), Complex(a,-b)]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
class SplitBiquaternion(Dyquaternion): pass

class Biquaternion(SquareQuadplex):
    units = ['','h','i','hi','j','hj','k','hk']
    subtype = Complex
    dftimag = 4
    def __init__(cxqn, *args):
        super().__init__(*args)
        if hasattr(cxqn, 'matrix'): return
        if len(args) == 2:
            a,b = args
            cxqn.matrix = type(cxqn)(a,0,b,0).matrix
        elif len(args) == 3:
            a,b,c = args
            cxqn.matrix = type(cxqn)(0,a,b,c).matrix
        elif len(args) == 4:
            a,aj,b,bj,c,cj,d,dj = _complex_split(*args)
            cxqn.matrix = type(cxqn)(a,aj,b,bj,c,cj,d,dj).matrix
        elif len(args) == 8:
            r,h,i,hi,j,hj,k,hk = _rationalized(*args)
            a,b,c,d = cxqn.subtype(r,h), cxqn.subtype(i,hi), cxqn.subtype(j,hj), cxqn.subtype(k,hk)
            cxqn.matrix = Matrix([[Complex(a,b), Complex(c,d)], [Complex(-c,d), Complex(a,-b)]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))

class DualQuaternion(SquareQuadplex):
    units = ['','i','j','k','ε','εi','εj','εk']
    subtype = Dual
    dftimag = 2
    def __init__(dlqn, *args):
        super().__init__(*args)
        if hasattr(dlqn, 'matrix'): return
        if len(args) == 2:
            a,b = args
            dlqn.matrix = type(dlqn)(a,0,b,0).matrix
        elif len(args) == 3:
            a,b,c = args
            dlqn.matrix = type(dlqn)(0,a,b,c).matrix
        elif len(args) == 4:
            a,b,c,d = args
            dlqn.matrix = type(dlqn)(a,b,c,d,0,0,0,0).matrix
        elif len(args) == 8:
            r,i,j,k,e,ei,ej,ek = _rationalized(*args)
            a,b,c,d = dlqn.subtype(r,e), dlqn.subtype(i,ei), dlqn.subtype(j,ej), dlqn.subtype(k,ek)
            dlqn.matrix = Matrix([[Complex(a,b), Complex(c,d)], [Complex(-c,d), Complex(a,-b)]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def elements(dlqn):
        elems = [x.elements() for x in dlqn.plex_elements()]
        return [elem[0] for elem in elems] + [elem[1] for elem in elems]
    def conjugate(dlqn):
        a,b,c,d = dlqn.plex_elements()
        elems = [x.elements() for x in [a,-b,-c,-d]]
        elems = [elem[0] for elem in elems] + [elem[1] for elem in elems]
        return type(dlqn)(elems)
    def plex_conjugate(dlqn):
        elems = [x.conjugate().elements() for x in dlqn.plex_elements()]
        elems = [elem[0] for elem in elems] + [elem[1] for elem in elems]
        return type(dlqn)(elems)

class QuartalQuaternion(SquareQuadplex):
    units = ['',  'hJ',   'hL', 'hJL',
             'i', 'ihJ', 'ihL', 'ihJL',
             'j', 'jhJ', 'jhL', 'jhJL',
             'k', 'khJ', 'khL', 'khJL']
    subtype = Quartal
    dftimag = 8
    def __init__(qtqn, *args):
        super().__init__(*args)
        if hasattr(qtqn, 'matrix'): return
        if len(args) == 2:
            a,b = args
            qtqn.matrix = type(qtqn)(a,0,b,0).matrix
        elif len(args) == 3:
            a,b,c = args
            qtqn.matrix = type(qtqn)(0,a,b,c).matrix
        elif len(args) == 4:
            a,b,c,d = args
            qtqn.matrix = type(qtqn)(a,0,b,0,c,0,d,0).matrix
        elif len(args) == 8:
            args = _complex_split(*args)
            qtqn.matrix = type(qtqn)(*args).matrix
        elif len(args) == 16:
            args = _rationalized(*args)
            a,b,c,d = [qtqn.subtype(args[x:x+4]) for x in range(0,16,4)]
            qtqn.matrix = Matrix([[Complex(a,b), Complex(c,d)], [Complex(-c,d), Complex(a,-b)]])
        else:
            raise ValueError("Incorrect number of arguments: %d" % len(args))
    def __invert__(qtqn):
        return type(qtqn)(qtqn.flatmatrix().inv())

