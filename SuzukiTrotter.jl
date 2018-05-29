@inline function getLength(x::Int64)
    return (63-leading_zeros(x))>>1
end


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

function normalizeDM!(rho)
    a=myTrace(rho)
    @. rho.=rho./a
end

@inline function triming(A; tol::Float64=2e-15)
    if abs(A)<tol
        return zero(typeof(A))
    else
        return A
    end
end

##########################################
###Building and reducing density matrix###
##########################################
#Build density matrix out of a vector. We assume the densityMatrix is dense for now
function buildDensityMatrix(psi)
    rho=zeros(Complex128, length(psi), length(psi));
    for j=1:length(psi)
        for k=1:length(psi)
            rho[j, k]=psi[j]*conj(psi[k])
        end
    end
    normalizeDM!(rho)
    return rho
end

###Building reduced DM from DM
###Convention : first site is zero
@inline function buildReducedDensityMatrix_small(rho, site)
    #Local densityMatrix for site site
    return transpose(At_mul_B(rho, TracesSmall[1+site, end-site]))
end

@inline function buildReducedDensityMatrix(rho, site, nb)    #Local densityMatrix for sites site:site+nb-1
    len=getLength(length(rho));
    if len==nb
        return rho
    end
    if nb==1
        return buildReducedDensityMatrix(rho, site)
    end
    return transpose(At_mul_B(rho, Traces[1+site, len-nb-site+1]))
end

@inline function buildReducedDensityMatrix_small(rho, site, nb)    #Local densityMatrix for sites site:site+nb-1
    len=getLength(length(rho));
    if len==nb
        return rho
    end
    if nb==1
        return buildReducedDensityMatrix_small(rho, site)
    end
    return transpose(At_mul_B(rho, TracesSmall[1+site, len-nb-site+1]))
end

@inline function reduceDensityMatrixLeft(rho)
    return transpose(At_mul_B(rho, Traces[2, 1]))
end

@inline function reduceDensityMatrixLeft!(redrho, rho)    #Local densityMatrix for site site
    redrho.=transpose(At_mul_B(rho, Traces[2, 1]))
end

@inline function reduceDensityMatrixRight(rho)
    return transpose(At_mul_B(rho, Traces[1, 2]))
end

@inline function reduceDensityMatrixRight!(redrho, rho)   #Local densityMatrix for site site
    redrho.=transpose(At_mul_B(rho, Traces[1, 2]))
end

@inline function reduceDensityMatrixLeft_small(rho)
    return transpose(At_mul_B(rho, TracesSmall[2, 1]))
end

@inline function reduceDensityMatrixLeft_small!(redrho, rho)    #Local densityMatrix for site site
    redrho.=transpose(At_mul_B(rho, TracesSmall[2, 1]))
end

@inline function reduceDensityMatrixRight_small(rho)
    return transpose(At_mul_B(rho, TracesSmall[1, 2]))
end

@inline function reduceDensityMatrixRight_small!(redrho, rho)   #Local densityMatrix for site site
    redrho.=transpose(At_mul_B(rho, TracesSmall[1, 2]))
end

#######################
###Entropy functions###
#######################
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

function computeEntropy(rho)  #Compute vonNeumannEE
    len=getLength(length(rho));
    return entropy(svdvals(reshape(rho, (2^len, 2^len))))
end

function mutualInformation(rho, start, nb1, nbtot)
    if nbtot==nb1
        return 0
    end
    len=getLength(length(rho));
    return computeEntropy(buildReducedDensityMatrix_small(rho, start+nb1, nbtot-nb1))+computeEntropy(buildReducedDensityMatrix_small(rho, start, nb1))-computeEntropy(buildReducedDensityMatrix_small(rho, start, nbtot))
end

####################
###Flattened kron###
####################
function leftFusionIdentity(rho);
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

###Much better
function leftFusionIdentity!(extrho, rho);
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

function leftFusion(smallrho, rho);
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


function leftFusion!(extrho, smallrho, rho);
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

function rightFusionIdentity(rho);
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


function rightFusionIdentity!(extrho, rho);
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

function rightFusion(rho, smallrho);
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

function rightFusion!(extrho, rho, smallrho);
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

    
############################################
###Buiding Hamiltonians, can be optimized###
############################################
function buildHamiltonian(hx, hz, maxLen)
    ham=spzeros(2^maxLen, 2^maxLen)
    ###Kinetic terms and sz
    for x=1:2^maxLen
        dig=2*digits(x-1, 2, maxLen)-1
        ###Kinetic terms and sz
        ham[x, x]=0.25*sum([dig[j]*dig[j+1] for j in 1:maxLen-1])-hz/2*(sum(dig)-0.5*dig[1]-0.5*dig[maxLen])
        for l=2:maxLen-1
            ham[x, x-dig[l]*2^(l-1)]+=hx/2
        end
        for l=[1, maxLen]
            ham[x, x-dig[l]*2^(l-1)]+=hx/2*0.5
        end
    end
    return ham
end

#Build Z_iZ_{i+1}- 0.5(Z_i+Z_{i+1})+0.5(X_i+X_{i+1})
function buildLocalHamiltonian(hx, hz, maxLen)
    dicHam=Dict()
    for site=1:maxLen-1
        dicHam[site]=spzeros(2^maxLen, 2^maxLen)
        for x=1:2^maxLen
            dig=2*digits(x-1, 2, maxLen)-1
            ###Kinetic terms and sz
            dicHam[site][x, x]=0.25*dig[site]*dig[site+1]-hz/2*(dig[site]+dig[site+1])/2
            dicHam[site][x, x-dig[site]*2^(site-1)]+=hx/2/2
            dicHam[site][x, x-dig[site+1]*2^(site)]+=hx/2/2
        end
    end
    return dicHam
end

function generateInitialState(maxLen)
    listRho=Array{Array{Complex128}}(maxLen)
    for l in 1:maxLen
        listRho[l]=zeros(Complex128, 2^maxLen, 2^maxLen)
    end
    for x in 1:2^maxLen
        dig=digits(x-1, 2, maxLen)
        for l=1:maxLen
            if dig[l]==1
                listRho[maxLen+1-l][x, x]=1/2^(maxLen-1)
            end
        end
    end
    return listRho
end


#############toDO
function computeEnergy()
    return 0
end

function joinDensityMatrix!(extendedRho, matRho, indexMin, indexMinTER, indexMax, indexMaxTER)
    if indexMin==indexMinTER && indexMax==indexMaxTER
        if mod(length(matRho), 2)==1
            println("Bug1 in joinDensityMatrix")
        end
        for l in 1:2:length(matRho)-1
            joinDensityMatAux!(extendedRho[(l+1)÷2], matRho[l], matRho[l+1])
        end
    elseif indexMin==indexMinTER && indexMax==indexMaxTER-1
        if mod(length(matRho), 2)==0
            println("Bug2 in joinDensityMatrix")
        end
        for l in 1:2:length(matRho)-1
            joinDensityMatAux!(extendedRho[(l+1)÷2], matRho[l], matRho[l+1])
        end
        rightFusionIdentity!(extendedRho[end], matRho[end])
    elseif indexMin==indexMinTER+1 && indexMax==indexMaxTER
        if mod(length(matRho), 2)==0
            println("Bug3 in joinDensityMatrix")
        end
        for l in 2:2:length(matRho)-1
            joinDensityMatAux!(extendedRho[l÷2+1], matRho[l], matRho[l+1])
        end
        leftFusionIdentity!(extendedRho[1], matRho[1])
    elseif indexMin==indexMinTER+1 && indexMax==indexMaxTER-1
        if mod(length(matRho), 2)==1
            println("Bug4 in joinDensityMatrix")
        end
        for l in 2:2:length(matRho)-1
            joinDensityMatAux!(extendedRho[l÷2+1], matRho[l], matRho[l+1])
        end
        leftFusionIdentity!(extendedRho[1], matRho[1])
        rightFusionIdentity!(extendedRho[end], matRho[end])
    else
        println("Bug in joinDensityMatrix!")
    end
end


function joinDensityMatAux!(rhoGuess, rhoLeft, rhoRight)
    len=getLength(length(rhoLeft));
    maxinconsistency=1e-8
    consistency=max(norm(reduceDensityMatrixRight(rhoGuess)-rhoLeft, 2), norm(reduceDensityMatrixLeft(rhoGuess)-rhoRight, 2)) ###Is it needed ?
    if consistency<maxinconsistency
        return 0
    end
    ###Check if input matrices are positive -> ignored
    ###See which product ansatz is best
    temprho1=buildReducedDensityMatrix_small(rhoLeft, 0)
    rho1=leftFusion(temprho1, reduceDensityMatrixRight_small(rhoRight))
    consistency1=norm(rho1-rhoLeft, 2)
    if consistency1<maxinconsistency
        rhoGuess.=leftFusion(temprho1, rhoRight)
        return 1
    end
    temprho1=buildReducedDensityMatrix_small(rhoRight, len-1)
    rho1=rightFusion(reduceDensityMatrixLeft_small(rhoLeft), temprho1)
    consistency1=norm(rho1-rhoRight, 2)
    if consistency1<maxinconsistency
        rhoGuess.=rightFusion(rhoLeft, temprho1)
        return 1
    end
    ###Fast method
    temp=transpose(At_mul_B(vcat(rhoLeft, rhoRight, 1), invMatJoin))
    temp-=transpose(At_mul_B(rhoGuess, pseudoIdentity))
    temp/=2
    tempconj=reshape(transpose(conj.(reshape(temp, (2^(len+1), 2^(len+1))))), length(temp))  ###A ameliorer
    for j=1:length(rhoGuess)
        rhoGuess[j]+=temp[j]+tempconj[j]
    end
    #if isposdef(rhoProposed)
    # consistency=max(norm(reduceDensityMatrixRight(rhoGuess)-rhoLeft, 2), norm(reduceDensityMatrixLeft(rhoGuess)-rhoRight, 2))
    # consistency=max(norm(reshape(reduceDensityMatrixRight(rhoGuess)-rhoLeft, (2^len, 2^len)), 1), norm(reshape(reduceDensityMatrixLeft(rhoGuess)-rhoRight, (2^len, 2^len)), 1))
    # 
    # #println(consistency)
     #if consistency<maxinconsistency
         return 2
    # end
    # #end 
     #println("Incomplete JoinDensityAux")
     #println(consistency)
    # return 3
end


###May be improved ?
# function specialMultLeft!(rho, UL)
#     lenmat=size(UL)[1];
#     rho.=reshape(UL*reshape(view(rho, :), (lenmat, lenmat)), lenmat*lenmat)
# end
# 
# 
# function specialMultRight!(rho, tUL)
#     lenmat=size(tUL)[1];
#     rho.=reshape(reshape(view(rho, :), (lenmat, lenmat))*tUL, lenmat*lenmat)
# end
# 
# function evolveMatrix!(extendedRho, ULeft, tULeft)
#     for l in 1:length(extendedRho)
#         specialMultLeft!(extendedRho[l], ULeft)
#         specialMultRight!(extendedRho[l], tULeft)
#     end
# end


function evolveMatrix!(extendedRho, ULeft, tULeft)
    lenmat=size(ULeft)[1];
    for l in 1:length(extendedRho)
        extendedRho[l]=reshape(ULeft*reshape(extendedRho[l], (lenmat, lenmat))*tULeft, lenmat*lenmat)
    end
end



function massReduce!(matRho, timeEvolvedRho, indexmin, indexminTER)
    if indexmin==indexminTER
        for l in 1:length(matRho)
            if mod(l, 2)==1
                reduceDensityMatrixRight!(matRho[l], timeEvolvedRho[(l+1)÷2])
            else
                reduceDensityMatrixLeft!(matRho[l], timeEvolvedRho[l÷2])
            end
            normalizeDM!(matRho[l])
        end
    elseif indexmin==indexminTER+1
        for l in 1:length(matRho)
            if mod(l, 2)==1
                reduceDensityMatrixLeft!(matRho[l], timeEvolvedRho[(l+1)÷2])
            else
                reduceDensityMatrixRight!(matRho[l], timeEvolvedRho[l÷2+1])
            end
            normalizeDM!(matRho[l])
        end    
    # elseif indexmin==indexminTER+2
    #     for l in 1:length(matRho)
    #         if mod(l, 2)==1
    #             reduceDensityMatrixRight!(matRho[l], timeEvolvedRho[(l+1)÷2+1])
    #         else
    #             reduceDensityMatrixLeft!(matRho[l], timeEvolvedRho[l÷2+1])
    #         end
    #     end
    else
        println("Bug in mass Reduce")
    end
end

function extendZone!(matRho, indexmin, indexmax, extendedCurrent, indicesCurrent, extendedOther, indicesOther)
    len=getLength(length(matRho[1]));
    ##Check if left is entangled
    mini=indexmin; maxi=indexmax;
    mi=mutualInformation(matRho[1], 0, 1, len)
    if mi>1e-7
        if indicesCurrent[1]==indexmin-1
            unshift!(matRho, reduceDensityMatrixRight(extendedCurrent[1]))
        elseif indicesCurrent[1]==indexmin
            unshift!(matRho, leftFusionIdentity(reduceDensityMatrixRight_small(matRho[1])))
            unshift!(extendedCurrent, leftFusionIdentity(matRho[1]))
            indicesCurrent[1]-=2
        else
            println("Bug in extendZone!")
        end
        mini-=1
        if indicesOther[1]==mini+1
            unshift!(extendedOther, leftFusionIdentity(matRho[1]))
            indicesOther[1]-=2
        elseif indicesOther[1]==mini+2
            println("Relou1 in extendZone")
            unshift!(extendedOther, leftFusionIdentity(leftFusionIdentity(reduceDensityMatrixRight_small(matRho[1]))))
            indicesOther[1]-=2
        end
    end
    ##Same for right extremity
    mi=mutualInformation(matRho[end], 0, len-1, len)
    if mi>1e-7
        if indicesCurrent[2]>indexmax
            push!(matRho, reduceDensityMatrixLeft(extendedCurrent[end]))
        elseif indicesCurrent[2]==indexmax
            push!(matRho, rightFusionIdentity(reduceDensityMatrixLeft_small(matRho[end])))
            push!(extendedCurrent, rightFusionIdentity(matRho[end]))
            indicesCurrent[2]+=2
        else
            println("Bug2 in extendZone!")
        end
        maxi+=1
        if indicesOther[2]==maxi-1
            push!(extendedOther, rightFusionIdentity(matRho[end]))
            indicesOther[2]+=2
        elseif indicesOther[2]==maxi-2
            println("Relou2 in extendZone")
            push!(extendedOther, rightFusionIdentity(rightFusionIdentity(reduceDensityMatrixLeft_small(matRho[end]))))
            indicesOther[2]+=2
        end
    end
    return mini, maxi
end

#############################################
###MatrixEquivalent of the trace operation###
#############################################
function buildTraceMatrixRight(maxLen)
    trmat=spzeros(2^(2*maxLen), 2^(2*maxLen+2))
    for x in 1:2^maxLen
        for y=1:2^maxLen
            trmat[x+(y-1)*2^maxLen, x+(y-1)*2^(maxLen+1)]=1
            trmat[x+(y-1)*2^maxLen, (x+2^maxLen)+(y-1+2^maxLen)*2^(maxLen+1)]=1
        end
    end
    return trmat
end


function buildTraceMatrixLeft(maxLen)
    trmat=spzeros(2^(2*maxLen), 2^(2*maxLen+2))
    for x in 1:2^maxLen
        for y=1:2^maxLen
            trmat[x+(y-1)*2^maxLen, 2*(x-1)+2*(y-1)*2^(maxLen+1)+1]=1
            trmat[x+(y-1)*2^maxLen, 1+2*(x-1)+(2*(y-1)+1)*2^(maxLen+1)+1]=1
        end
    end
    return trmat
end


function buildTraceMatrix(maxLen)
    trmat=spzeros(1, 2^(2*maxLen+2))
    for x in 1:2^(maxLen+1)
        trmat[x+2^(maxLen+1)*(x-1)]=1
    end
    return trmat
end


function main(hx::Float64, hz::Float64, T::Float64; maxLen=3, dt=0.0001)
    #Define Pauli Matrices
    sx=[0 1; 1 0]; sz=[1  0 ; 0 -1]; sy=[0 -1im; 1im 0]
    tim=dt:dt:T
    #Define Hamiltonian
    hamiltonian=buildHamiltonian(hx, hz, maxLen)
    dicLocalHamiltonianInside=buildLocalHamiltonian(hx, hz, maxLen)  #### Build local Hamiltonian i:i+1 for i=0:maxLen-1 for maxLen+1 sites matrices. 
    dicLocalHamiltonian=buildLocalHamiltonian(hx, hz, maxLen+1)  #### Build local Hamiltonian i:i+1 for i=0:maxLen-1 for maxLen+1 sites matrices. 
    ###Evolution operator for even and odd sites
    ###toDo: reexpress as evolution on vectors
    ULeft=expm(full(-1im*dt*sum([dicLocalHamiltonian[x] for x in 1:2:maxLen+1])))
    tULeft=ULeft'
    #StockedOperator
    extendedRho_o=Array{Array{Complex128, 1}}((maxLen+1)÷2); extendedRho_e=Array{Array{Complex128, 1}}((maxLen+1)÷2);
    for l in 1:length(extendedRho_e)
        extendedRho_o[l]=reshape(eye(2^(maxLen+1), 2^(maxLen+1))/2^(maxLen+1),  2^(maxLen*2+2))
        extendedRho_e[l]=reshape(eye(2^(maxLen+1), 2^(maxLen+1))/2^(maxLen+1),  2^(maxLen*2+2))
    end
    ###Setting up the global trace operators: note, the original one site reduced DM is marginally faster
    ###Choosing global matrixes seem to slow down code by ~1% but is much easier to read. More benchmark to be done later ?
    if !isdefined(:TrLeft)
        global const TrRight=buildTraceMatrixRight(maxLen);
        global const Tr=buildTraceMatrix(maxLen);
        global const TrLeft=buildTraceMatrixLeft(maxLen);
        global const Trsmall=buildTraceMatrix(maxLen-1);
        global const MatJoin=transpose(sparse(vcat(TrRight, TrLeft, Tr)));
        global const invMatJoin=sparse(triming.(pinv(full(MatJoin))));
        global const pseudoIdentity=sparse(triming.(MatJoin*invMatJoin));
        tempTraces=Array{typeof(TrLeft)}(maxLen+1, maxLen+1)
        tempTraces[1, 1]=speye(2^(2*maxLen+2), 2^(2*maxLen+2));
        for j=2:maxLen+1
            tempTraces[j, 1]=transpose(buildTraceMatrixLeft(maxLen-j+2)*transpose(tempTraces[j-1, 1]))
        end
        for j=1:maxLen
            for k=2:maxLen+2-j
                tempTraces[j, k]=transpose(buildTraceMatrixRight(maxLen-j-k+3)*transpose(tempTraces[j, k-1]))
            end
        end
        global const Traces=tempTraces
        tempTraces=Array{typeof(TrLeft)}(maxLen, maxLen)
        tempTraces[1, 1]=speye(2^(2*maxLen), 2^(2*maxLen));
        for j=2:maxLen
            tempTraces[j, 1]=transpose(buildTraceMatrixLeft(maxLen-j+1)*transpose(tempTraces[j-1, 1]))
        end
        for j=1:maxLen-1
            for k=2:maxLen+1-j
                tempTraces[j, k]=transpose(buildTraceMatrixRight(maxLen-j-k+2)*transpose(tempTraces[j, k-1]))
            end
        end
        global const TracesSmall=tempTraces
    end
    #vectKernelMatJoin=triming.(nullspace(full(MatJoin)));
    ###Observables###
    tim2=0:100*dt:T;
    szObservable=zeros((length(tim)+99)÷100+1);
    sxObservable=zeros((length(tim)+99)÷100+1);
    #initializeState
    matRho=full.(reshape.(generateInitialState(maxLen), 2^(2*maxLen)))  #Random state with 0 site projected on +1 => DM [-1, 0 1]. Stored in vector form.
    minMatRho=-maxLen+1; maxMatRho=maxLen-1
    if mod(minMatRho, 2)==0
        minTER_e=minMatRho; minTER_o=minMatRho-1
    else
        minTER_e=minMatRho-1; minTER_o=minMatRho
    end
    maxTER_e=minTER_e+2*(length(extendedRho_e)-1)+maxLen; maxTER_o=minTER_o+2*(length(extendedRho_o)-1)+maxLen
    indices_e=[minTER_e, maxTER_e]; indices_o=[minTER_o, maxTER_o]
    #toDO: a first few good steps
    #timeEvolution: evolution even then evolution odd
    tic()
    cnt=0; totaltime=0; cnt2=2;
    szObservable[1]=real(trace(sz*reshape(buildReducedDensityMatrix_small(matRho[1-minMatRho], 0), (2, 2))))
    sxObservable[1]=real(trace(sx*reshape(buildReducedDensityMatrix_small(matRho[1-minMatRho], 0), (2, 2))))
    for j in 1:length(tim)
        ####Convention: apply Odd then Even
        if mod(j, (length(tim)+99)÷100)==0
            cnt+=1
            ctime=toq()
            println(string(cnt)*"% of simulation done. Time for the last %: "*string(round(ctime, 2))*"sec.")
            totaltime+=ctime
            println("Total elapsed time: "*string(round(totaltime, 2))*"sec.")
            println((minMatRho, maxMatRho))
            tic()
        end
        if tim[j]==tim2[cnt2]
            szObservable[cnt2]=real(trace(sz*reshape(buildReducedDensityMatrix_small(matRho[1-minMatRho], 0), (2, 2))))
            sxObservable[cnt2]=real(trace(sx*reshape(buildReducedDensityMatrix_small(matRho[1-minMatRho], 0), (2, 2))))
            cnt2+=1
        end
        joinDensityMatrix!(extendedRho_o, matRho, minMatRho, indices_o[1], maxMatRho, indices_o[2]);   #Build the larger density Matrix
        evolveMatrix!(extendedRho_o, ULeft, tULeft);   #Evolve the larger densityMatrix
        massReduce!(matRho, extendedRho_o, minMatRho, indices_o[1]) #Compute the update densityMatrix
        minMatRho, maxMatRho=extendZone!(matRho, minMatRho, maxMatRho, extendedRho_o, indices_o, extendedRho_e, indices_e)
        joinDensityMatrix!(extendedRho_e, matRho, minMatRho, indices_e[1], maxMatRho, indices_e[2])   #Build the larger density Matrix
        evolveMatrix!(extendedRho_e, ULeft, tULeft)
        massReduce!(matRho, extendedRho_e, minMatRho, indices_e[1])
        minMatRho, maxMatRho=extendZone!(matRho, minMatRho, maxMatRho, extendedRho_e, indices_e, extendedRho_o, indices_o)
        ##Measure sz
    end
    toc()
    return tim2, szObservable, sxObservable
end
    
    
###ToDo: remplacer isposdef par issemiposdef
###ToDo: optimize matrix product in joinDensity ? Have an explicit function compiled probably.



    
    
