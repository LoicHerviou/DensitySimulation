#############################################
###MatrixEquivalent of the trace operation###
#############################################
"""
    buildTraceMatrixRight(maxLen; typed=Float64)

Create the matrix that takes the trace on the right-most site of a DM of length maxLen
typed is the type of the data store in the matrix: for rational arythmetic, choose SymPy.Sym
"""
function buildTraceMatrixRight(maxLen; typed=Float64)
    trmat=spzeros(typed, 2^(2*maxLen), 2^(2*maxLen+2))
    for x in 1:2^maxLen
        for y=1:2^maxLen
            trmat[x+(y-1)*2^maxLen, x+(y-1)*2^(maxLen+1)]=1
            trmat[x+(y-1)*2^maxLen, (x+2^maxLen)+(y-1+2^maxLen)*2^(maxLen+1)]=1
        end
    end
    return trmat
end
"""
    buildTraceMatrixLeft(maxLen; typed=Float64)

create the matrix that takes the trace on the left-most site of a DM of length maxLen
typed is the type of the data store in the matrix: for rational arythmetic, choose SymPy.Sym
"""
function buildTraceMatrixLeft(maxLen; typed=Float64)
    trmat=spzeros(typed, 2^(2*maxLen), 2^(2*maxLen+2))
    for x in 1:2^maxLen
        for y=1:2^maxLen
            trmat[x+(y-1)*2^maxLen, 2*(x-1)+2*(y-1)*2^(maxLen+1)+1]=1
            trmat[x+(y-1)*2^maxLen, 1+2*(x-1)+(2*(y-1)+1)*2^(maxLen+1)+1]=1
        end
    end
    return trmat
end

"""
    buildTraceMatrix(maxLen)

create the matrix that takes the trace of a DM of length maxLen
"""
function buildTraceMatrix(maxLen)
    trmat=spzeros(1, 2^(2*maxLen+2))
    for x in 1:2^(maxLen+1)
        trmat[x+2^(maxLen+1)*(x-1)]=1
    end
    return trmat
end

"""
    initializeTraceMatrices(maxLen)

Build a complete table of Traces
Traces[i, j] takes (i-1) the trace on the left, and (j-1) the trace on the right
"""
function initializeTraceMatrices(maxLen)
    TrLeft=buildTraceMatrixLeft(2);
    #global const pseudoIdentity=sparse(triming.(MatJoin*invMatJoin));
    Traces=Array{typeof(TrLeft)}(maxLen, maxLen)
    Traces[1, 1]=speye(2^(2*maxLen), 2^(2*maxLen));
    for j=2:maxLen
        Traces[j, 1]=transpose(buildTraceMatrixLeft(maxLen-j+1)*transpose(Traces[j-1, 1]))
    end
    for j=1:maxLen-1
        for k=2:maxLen+1-j
            Traces[j, k]=transpose(buildTraceMatrixRight(maxLen-j-k+2)*transpose(Traces[j, k-1]))
        end
    end
    return Traces
end
##########################################
###Building and reducing density matrix###
##########################################
"""
    buildReducedDensityMatrix(rho, site, Traces)

Compute the single-site density matrix of rho on site site.
Convention site=0 is the leftmost site.
Traces is the set of Trace matrices previously computed
"""
@inline function buildReducedDensityMatrix(rho, site, Traces)
    return transpose(At_mul_B(rho, Traces[1+site, end-site]))
end

"""
    buildReducedDensityMatrix(rho, site, nb, Traces)

Compute the reduced density matrix of rho on sites site:site+nb-1.
Convention site=0 is the leftmost site.
Traces is the set of Trace matrices previously computed
"""
@inline function buildReducedDensityMatrix(rho, site, nb, Traces)    #Local densityMatrix for sites site:site+nb-1
    len=getLength(length(rho));
    if len==nb
        return rho
    end
    if nb==1
        return buildReducedDensityMatrix(rho, site, Traces)
    end
    return transpose(At_mul_B(rho, Traces[1+site, len-nb-site+1]))
end

"""
    reduceDensityMatrixLeft(rho, Traces)

Takes the left trace of rho
Traces is the set of Trace matrices previously computed
"""
@inline function reduceDensityMatrixLeft(rho, Traces)
    return transpose(At_mul_B(rho, Traces[2, 1]))
end

"""
    reduceDensityMatrixLeft!(redrho, rho, Traces)

Takes the left trace of rho to redrho
Traces is the set of Trace matrices previously computed
"""
@inline function reduceDensityMatrixLeft!(redrho, rho, Traces)    #Local densityMatrix for site site
    redrho.=transpose(At_mul_B(rho, Traces[2, 1]))
end

"""
    reduceDensityMatrixRight(rho, Traces)

Takes the right trace of rho
Traces is the set of Trace matrices previously computed
"""
@inline function reduceDensityMatrixRight(rho, Traces)
    return transpose(At_mul_B(rho, Traces[1, 2]))
end

"""
    reduceDensityMatrixRight!(redrho, rho, Traces)

Takes the right trace of rho to redrho
Traces is the set of Trace matrices previously computed
"""
@inline function reduceDensityMatrixRight!(redrho, rho, Traces)   #Local densityMatrix for site site
    redrho.=transpose(At_mul_B(rho, Traces[1, 2]))
end


####################
###Reimplementation of kron###
####################
"""
    leftFusionIdentity(rho)

Compute I2⊗rho in vector form
Note that it would correspond to kron(rho, I2) for matrices
"""
function leftFusionIdentity(rho)
        len=getLength(length(rho));
        stepp=2^len
        extrho=zeros(Complex128, 2^(2*len+2))
        offset=0; offset2=0
        for j=1:stepp
            for k=1:stepp
                @inbounds extrho[offset2+2*k-1]=0.5*rho[offset+k]
                @inbounds extrho[offset2+2*stepp+2*k]=0.5*rho[offset+k]
            end
            offset+=stepp
            offset2+=4*stepp
        end
        return extrho
end

"""
    leftFusionIdentity(extrho, rho)

Compute extrho=I2⊗rho in vector form
Note that it would correspond to kron(rho, I2) for matrices
"""
function leftFusionIdentity!(extrho, rho)
        len=getLength(length(rho));
        @simd for i =1:length(extrho)
            @inbounds extrho[i]=zero(Complex128)
        end
        stepp=2^len
        offset=0; offset2=0
        for j=1:stepp
            for k=1:stepp
                @inbounds extrho[offset2+2*k-1]=0.5*rho[offset+k]
                @inbounds extrho[offset2+2*stepp+2*k]=0.5*rho[offset+k]
            end
            offset+=stepp
            offset2+=4*stepp
        end
end

"""
    leftFusion(smallrho, rho)

Compute smallrho⊗rho in vector form
Note that it would correspond to kron(rho, smallrho) for matrices
"""
function leftFusion(smallrho, rho)
        len=getLength(length(rho));
        extrho=Array{Complex128}(2^(2*len+2))
        stepp=2^len
        offset=0; offset2=0
        for j=1:stepp
            for k=1:stepp
                @inbounds extrho[offset2+2*k-1]=smallrho[1]*rho[offset+k]
                @inbounds extrho[offset2+2*k]=smallrho[2]*rho[offset+k]
                @inbounds extrho[offset2+2*stepp+2*k-1]=smallrho[3]*rho[offset+k]
                @inbounds extrho[offset2+2*stepp+2*k]=smallrho[4]*rho[offset+k]
            end
            offset+=stepp
            offset2+=4*stepp
        end
        return extrho
end

"""
    leftFusion!(extrho, smallrho, rho)

Compute extrho=smallrho⊗rho in vector form
Note that it would correspond to kron(rho, smallrho) for matrices
"""
function leftFusion!(extrho, smallrho, rho)
        len=getLength(length(rho));
        stepp=2^len
        offset=0; offset2=0
        for j=1:stepp
            for k=1:stepp
                @inbounds extrho[offset2+2*k-1]=smallrho[1]*rho[offset+k]
                @inbounds extrho[offset2+2*k]=smallrho[2]*rho[offset+k]
                @inbounds extrho[offset2+2*stepp+2*k-1]=smallrho[3]*rho[offset+k]
                @inbounds extrho[offset2+2*stepp+2*k]=smallrho[4]*rho[offset+k]
            end
            offset+=stepp
            offset2+=4*stepp
        end
end

"""
    rightFusionIdentity(rho)

Compute rho⊗I2 in vector form
Note that it would correspond to kron(I2, rho) for matrices
"""
function rightFusionIdentity(rho)
        len=getLength(length(rho));
        stepp=2^len; temp=2^(2*len+1)+stepp
        extrho=zeros(Complex128, 2^(2*len+2))
        offset=0; offset2=0
        for j=1:stepp
            for k=1:stepp
                @inbounds extrho[offset2+k]=0.5*rho[offset+k]
                @inbounds extrho[offset2+k+temp]=0.5*rho[offset+k]
            end
            offset+=stepp
            offset2+=2*stepp
        end
        return extrho
end

"""
    rightFusionIdentity!(extrho, rho)

Compute extrho=rho⊗I2 in vector form
Note that it would correspond to kron(I2, rho) for matrices
"""
function rightFusionIdentity!(extrho, rho)
        len=getLength(length(rho));
        stepp=2^len; temp=2^(2*len+1)+stepp
        @simd for i=1:length(extrho)
            @inbounds extrho[i]=zero(Complex128)
        end
        offset=0; offset2=0
        for j=1:stepp
            for k=1:stepp
                @inbounds extrho[offset2+k]=0.5*rho[offset+k]
                @inbounds extrho[offset2+k+temp]=0.5*rho[offset+k]
            end
            offset+=stepp
            offset2+=2*stepp
        end
end

"""
    rightFusion(rho, smallrho)

Compute rho⊗smallrho in vector form
Note that it would correspond to kron(smallrho, rho) for matrices
"""
function rightFusion(rho, smallrho)
    len=getLength(length(rho));
    stepp=2^len; temp=2^(2*len+1)
    extrho=Array{Complex128}(2^(2*len+2))
    offset=0; offset2=0
    for j=1:stepp
        for k=1:stepp
            @inbounds extrho[offset2+k]=smallrho[1]*rho[offset+k]
            @inbounds extrho[offset2+k+stepp]=smallrho[2]*rho[offset+k]
            @inbounds extrho[offset2+k+temp]=smallrho[3]*rho[offset+k]
            @inbounds extrho[offset2+k+temp+stepp]=smallrho[4]*rho[offset+k]
        end
        offset+=stepp
        offset2+=2*stepp
    end
    return extrho
end

"""
    rightFusion!(extrho, rho, smallrho);

Compute extrho=rho⊗smallrho in vector form
Note that it would correspond to kron(smallrho, rho) for matrices
"""
function rightFusion!(extrho, rho, smallrho)
    len=getLength(length(rho));
    stepp=2^len; temp=2^(2*len+1)
    offset=0; offset2=0
    for j=1:stepp
        for k=1:stepp
            @inbounds extrho[offset2+k]=smallrho[1]*rho[offset+k]
            @inbounds extrho[offset2+k+stepp]=smallrho[2]*rho[offset+k]
            @inbounds extrho[offset2+k+temp]=smallrho[3]*rho[offset+k]
            @inbounds extrho[offset2+k+temp+stepp]=smallrho[4]*rho[offset+k]
        end
        offset+=stepp
        offset2+=2*stepp
    end
end

########################################################################################
###Build from an operator acting on a DM the corresponding superoperator for a vector###
########################################################################################
"""
    buildSuperOperatorLeft(operator)

Return the sparse matrix representing operator acting on the left on a DM written as a vector
"""
function buildSuperOperatorLeft(operator)
    return kron(speye(size(operator)[1]), operator)''
end

"""
    buildSuperOperatorRight(operator)

Return the sparse matrix representing operator acting on the right on a DM written as a vector
"""
function buildSuperOperatorRight(operator)
    return kron(transpose(operator), speye(size(operator)[1]))''
end

"""
    buildSuperCommutator(operator)

Return the sparse matrix representing [operator, ρ] for ρ written as a vector
"""
function buildSuperCommutator(operator)
    return (buildSuperOperatorLeft(operator)-buildSuperOperatorRight(operator))''
end
