###################
###Miscellaneous###
###################
"""
    getLength(x::Int64)

Compute the physical length of a density matrix from the length x of its vector representation.
"""
@inline function getLength(x::Int64)
    return (63-leading_zeros(x))>>1
end

"""
    triming!(A; tol::Float64=2e-15)

Put to 0 all the elements of A smaller in absolute value than tol.
"""
@inline function triming!(A; tol::Float64=2e-15)
    for x in 1:length(A)
        if abs(A[x])<tol
            A[x]=zero(typeof(A[x]))
        end
    end
end
"""
    triming(x; tol::Float64=2e-15)

Return 0 if x is smaller than tol, x else
"""
@inline function triming(x; tol::Float64=2e-15)
    if abs(x)<tol
        return zero(typeof(x))
    else
        return x
    end
end
#######################
###Entropy functions###
#######################
"""
    entropy(eigen)

From a list of svs, return the vN entropy. sv should be real and between 0 and 1
"""
@inline function entropy(eigen)
    entr=0.
    @simd for x in eigen
        #println(x)
        if 1.>x>0.
            #entr-=real(x)*log(real(x))
            entr-=x*log(x)
        end
    end
    return entr
end

"""
    computeEntropy(rho)

Compute the vN entanglement entropy of the density matrix rho, stored as a vector
"""
@inline function computeEntropy(rho)  #Compute vonNeumannEE
    len=getLength(length(rho));
    return entropy(svdvals(reshape(rho, (2^len, 2^len))))
end

"""
    mutualInformation(rho, start, nb1, nbtot)

Compute the mutual Information between start:start+nb1-1 and start+nb1:start+nbtot-1
"""
function mutualInformation(rho, start, nb1, nbtot, Traces)
    if nbtot==nb1
        return 0
    end
    len=getLength(length(rho));
    return computeEntropy(buildReducedDensityMatrix(rho, start+nb1, nbtot-nb1, Traces))+computeEntropy(buildReducedDensityMatrix(rho, start, nb1, Traces))-computeEntropy(buildReducedDensityMatrix(rho, start, nbtot, Traces))
end

"""
    myTrace(rho)

Compute the trace of a DM while removing the imaginary part of diag term
"""
@inline function myTrace(rho)
    tr=0.0
    len=getLength(length(rho));
    stepp=2^len+1; offset=1
    for j=1:2^len
        @inbounds rho[offset]=rho[offset].re
        @inbounds tr+=rho[offset].re
        offset+=stepp
    end
    return tr
end

"""
    normalizeDM!(rho)

Normalize a DM stocked as a vector by its trace
"""
function normalizeDM!(rho)
    a=myTrace(rho)
    @. rho.=rho./a
end
