@inline function getLength(x::Int64)
    return (63-leading_zeros(x))>>1
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
###Building reduced DM from DM
###Convention : first site is zero
@inline function buildReducedDensityMatrix(rho, site)
    #Local densityMatrix for site site
    return transpose(At_mul_B(rho, Traces[1+site, end-site]))
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
    return computeEntropy(buildReducedDensityMatrix(rho, start+nb1, nbtot-nb1))+computeEntropy(buildReducedDensityMatrix(rho, start, nb1))-computeEntropy(buildReducedDensityMatrix(rho, start, nbtot))
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
#Build X_iX_{i+1}+hz  Z_i+ hxX_i
function buildInsideHamiltonian(hx, hz, maxLen)
    dicHam=spzeros(2^maxLen, 2^maxLen)
    for x=1:2^maxLen
        dig=2*digits(x-1, 2, maxLen)-1
        for site=1:maxLen-1
            dicHam[x, x-dig[site]*2^(site-1)-dig[site+1]*2^(site)]+=0.25
        end
        for site=1:maxLen
            dicHam[x, x]+=hz/2*dig[site]
            dicHam[x, x-dig[site]*2^(site-1)]+=hx/2
        end
    end
    return dicHam
end

function buildLinkHamiltonian(hx, hz, maxLen)
    dicHam=Array{typeof(spzeros(2,2))}(2)
    dicHam[1]=spzeros(2^maxLen, 2^maxLen);  dicHam[2]=spzeros(2^maxLen, 2^maxLen)
    for x=1:2^maxLen
        dig=2*digits(x-1, 2, maxLen)-1
        dicHam[1][x, x-dig[1]-dig[2]*2]+=0.25
        dicHam[2][x, x-dig[end-1]*2^(maxLen-2)-dig[end]*2^(maxLen-1)]+=0.25
    end
    return dicHam
end



function generateInitialState_z(maxLen)
    listRho=Array{Array{Complex128}}(maxLen)
    localRho=[0.0+0.0im 0.0; 0.0 1.0]
    for l in 1:maxLen
        listRho[l]=kron(kron(eye(Complex128, 2^(l-1)),  localRho), eye(Complex128, 2^(maxLen-l)))/2^(maxLen-1)
    end
    return listRho
end


function generateInitialState_x(maxLen)
    listRho=Array{Array{Complex128}}(maxLen)
    localRho=[0.5+0.0im 0.5; 0.5 0.5]
    for l in 1:maxLen
        listRho[l]=kron(kron(eye(Complex128, 2^(l-1)),  localRho), eye(Complex128, 2^(maxLen-l)))/2^(maxLen-1)
    end
    return listRho
end

function buildSuperOperatorLeft(operator)
    return kron(speye(size(operator)[1]), operator)''
end

function buildSuperOperatorRight(operator)
    return kron(transpose(operator), speye(size(operator)[1]))''
end


function buildSuperCommutator(operator)
    return (buildSuperOperatorLeft(operator)-buildSuperOperatorRight(operator))''
end



#############toDO
function computeEnergy()
    return 0
end

###########################
###Core of the algorithm###
###########################
function computeDerivative!(vectdRho, vectRho, Hinside, leftApprox, rightApprox, tempMatrix)
    insideEvolution!(vectdRho, vectRho, Hinside)
    outsideEvolution!(vectdRho, vectRho, leftApprox, rightApprox, tempMatrix)
end

function insideEvolution!(vectDRho, vectRho, Hinside)
    for j=1:length(vectRho)
        A_mul_B!(vectDRho[j], Hinside, vectRho[j])
    end
end

function outsideEvolution!(vectdRho, vectRho, leftApprox, rightApprox, tempMatrix)
    ###Auxiliary matrix to stock the boundaries  toDO=>removeThatStep
    leftFusionIdentity!(tempMatrix[1], reduceDensityMatrixRight(vectRho[1]));
    rightFusionIdentity!(tempMatrix[2], reduceDensityMatrixLeft(vectRho[end]));
    ###
    for j=1:length(vectdRho)-1
        vectdRho[j].=vectdRho[j]+rightApprox*vectRho[j+1]
    end
    vectdRho[end].=vectdRho[end]+rightApprox*tempMatrix[2]
    for j=2:length(vectdRho)
        vectdRho[j].=vectdRho[j]+leftApprox*vectRho[j-1]
    end
    vectdRho[1].=vectdRho[1]+leftApprox*tempMatrix[1]
end

function extendZone!(vectDRho, vectRho)
    len=getLength(length(vectRho[1])); expanded=false
    ##Check if left is entangled
    mi=mutualInformation(vectRho[1], 0, 1, len)
    if mi>1e-7
        unshift!(vectRho, leftFusionIdentity(reduceDensityMatrixRight(vectRho[1])))
        unshift!(vectDRho, zeros(Complex128, length(vectDRho[1])))
        expanded=true
    end
    ##Same for right extremity
    mi=mutualInformation(vectRho[end], 0, len-1, len)
    if mi>1e-7
        push!(vectRho, rightFusionIdentity(reduceDensityMatrixLeft(vectRho[end])))
        push!(vectDRho, zeros(Complex128, length(vectDRho[1])))
    end
    if expanded
        return 1
    else
        return 0
    end
end


function extendZone!(vectDRho, vectDRhoAux, vectRho)
    len=getLength(length(vectRho[1])); expanded=false
    ##Check if left is entangled
    mi=mutualInformation(vectRho[1], 0, 1, len)
    if mi>1e-7
        unshift!(vectRho, leftFusionIdentity(reduceDensityMatrixRight(vectRho[1])))
        unshift!(vectDRho, zeros(Complex128, length(vectDRho[1])))
        unshift!(vectDRhoAux, zeros(Complex128, length(vectDRhoAux[1])))
        expanded=true
    end
    ##Same for right extremity
    mi=mutualInformation(vectRho[end], 0, len-1, len)
    if mi>1e-7
        push!(vectRho, rightFusionIdentity(reduceDensityMatrixLeft(vectRho[end])))
        push!(vectDRho, zeros(Complex128, length(vectDRho[1])))
        push!(vectDRhoAux, zeros(Complex128, length(vectDRhoAux[1])))
    end
    if expanded
        return 1
    else
        return 0
    end
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


function initializeTraceMatrices(maxLen, Hlink, Hcommuting)
    if !isdefined(:TrLeft)
        global const TrRight=buildTraceMatrixRight(maxLen-1);
        global const Tr=buildTraceMatrix(maxLen-1);
        global const TrLeft=buildTraceMatrixLeft(maxLen-1);
        #global const pseudoIdentity=sparse(triming.(MatJoin*invMatJoin));
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
        global const Traces=tempTraces
    end
    ###For the left link
    leftCommuter=buildSuperOperatorLeft(kron(speye(2^(maxLen-1)), Hcommuting))
    MatJoin=sparse(vcat(TrRight, TrLeft, TrLeft*leftCommuter));
    invMatJoin=sparse(triming.(pinv(full(MatJoin))));
    leftEvolutionOperator=invMatJoin[:, 1:2^(2*maxLen-2)]*TrLeft*Hlink[1]
    ###For the right link
    rightCommuter=buildSuperOperatorLeft(kron(Hcommuting, speye(2^(maxLen-1))))
    MatJoin=sparse(vcat(TrLeft, TrRight, TrRight*rightCommuter));
    invMatJoin=sparse(triming.(pinv(full(MatJoin))));
    rightEvolutionOperator=invMatJoin[:, 1:2^(2*maxLen-2)]*TrRight*Hlink[2]
    return leftEvolutionOperator, rightEvolutionOperator
end

#
# function initializeTraceMatrices(maxLen, Hlink, Hcommuting)
#     if !isdefined(:TrLeft)
#         global const TrRight=buildTraceMatrixRight(maxLen-1);
#         global const Tr=buildTraceMatrix(maxLen-1);
#         global const TrLeft=buildTraceMatrixLeft(maxLen-1);
#         #global const pseudoIdentity=sparse(triming.(MatJoin*invMatJoin));
#         tempTraces=Array{typeof(TrLeft)}(maxLen, maxLen)
#         tempTraces[1, 1]=speye(2^(2*maxLen), 2^(2*maxLen));
#         for j=2:maxLen
#             tempTraces[j, 1]=transpose(buildTraceMatrixLeft(maxLen-j+1)*transpose(tempTraces[j-1, 1]))
#         end
#         for j=1:maxLen-1
#             for k=2:maxLen+1-j
#                 tempTraces[j, k]=transpose(buildTraceMatrixRight(maxLen-j-k+2)*transpose(tempTraces[j, k-1]))
#             end
#         end
#         global const Traces=tempTraces
#     end
#     ###For the left link
#     leftCommuter=buildSuperOperatorLeft(kron(speye(2^(maxLen-1)), Hcommuting))
#     MatJoin=sparse(vcat(TrRight, TrLeft, TrLeft*leftCommuter));
#     tempLeftEvolutionOperator=speye(size(MatJoin)[1])[:, 1:2^(2*maxLen-2)]
#     leftEvolutionOperator=spzeros(size(MatJoin)[2], 2^(2*maxLen-2))
#     for j=1:2^(2*maxLen-2)
#         leftEvolutionOperator[:, j]=triming.(MatJoin\full(tempLeftEvolutionOperator[:, j]))
#     end
#     leftEvolutionOperator=leftEvolutionOperator*TrLeft*Hlink[1]
#     ###For the right link
#     rightCommuter=buildSuperOperatorLeft(kron(Hcommuting, speye(2^(maxLen-1))))
#     MatJoin=sparse(vcat(TrLeft, TrRight, TrRight*rightCommuter));
#     tempRightEvolutionOperator=speye(size(MatJoin)[1])[:, 1:2^(2*maxLen-2)]
#     rightEvolutionOperator=spzeros(size(MatJoin)[2], 2^(2*maxLen-2))
#     for j=1:2^(2*maxLen-2)
#         rightEvolutionOperator[:, j]=triming.(MatJoin\full(tempLeftEvolutionOperator[:, j]))
#     end
#     rightEvolutionOperator=rightEvolutionOperator*TrRight*Hlink[2]
#     return leftEvolutionOperator, rightEvolutionOperator
# end
#

###toDo look static arrays ?
function main(hx::Float64, hz::Float64, T::Float64; maxLen=3, dt=0.01, order=2)
    #Define Pauli Matrices
    sx=[0 1; 1 0]; sz=-[1  0 ; 0 -1]; sy=[0 -1im; 1im 0]
    tim=0:dt:T
    #Define Hamiltonian
    Hinside=buildSuperCommutator(-1im*dt*buildInsideHamiltonian(hx, hz, maxLen));  #### Always multiplication on the right => slightly faster. Will have to recheck
    Hlink=buildSuperCommutator.(-1im*dt*buildLinkHamiltonian(hx, hz, maxLen))
    ###Setting up the global trace operators: note, the original one site reduced DM is marginally faster
    ###Choosing global matrixes seem to slow down code by ~1% but is much easier to read. More benchmark to be done later ?
    leftApproximation, rightApproximation=initializeTraceMatrices(maxLen, Hlink, sparse(sz))
    ###Initialization of the density operators.
    vectRho=map(x->reshape(x, length(x)), generateInitialState_z(maxLen));  minMatRho=-maxLen+1
    tempMatrix=[zeros(Complex128, 2^(2*maxLen)), zeros(Complex128, 2^(2*maxLen))]
    if order==1
        vectDRho=similar(vectRho)
        for j=1:length(vectDRho)
            vectDRho[j]=zeros(Complex128, 2^(2*maxLen))
        end
    elseif order==2
        vectDRho=similar(vectRho);  vectDRhoAux=similar(vectRho)
        for j=1:length(vectDRho)
            vectDRho[j]=zeros(Complex128, 2^(2*maxLen))
            vectDRhoAux[j]=zeros(Complex128, 2^(2*maxLen))
        end
    elseif order==4
        vectDRho=similar(vectRho);  vectDRhoAux=similar(vectRho)
        for j=1:length(vectDRho)
            vectDRho[j]=zeros(Complex128, 2^(2*maxLen))
            vectDRhoAux[j]=zeros(Complex128, 2^(2*maxLen))
        end
    else
        println("This order has not been implemented")
        return nothing
    end
    ###Observables
    tim2=0:10*dt:T;
    szObservable=zeros(length(tim2));
    sxObservable=zeros(length(tim2));
    #test=zeros(length(tim));
    ###Last preparations
    tic()
    cnt=0; totaltime=0; cnt2=1;
    #szObservable[1]=real(trace(sz*reshape(buildReducedDensityMatrix_small(vectRho[1-minMatRho], 0), (2, 2))))
    #sxObservable[1]=real(trace(sx*reshape(buildReducedDensityMatrix_small(vectRho[1-minMatRho], 0), (2, 2))))
    ###Start of the time evolution
    for j in 1:length(tim)
        ####Printing update
        if mod(j, (length(tim)+99)÷100)==0
            cnt+=1
            ctime=toq()
            println(string(cnt)*"% of simulation done. Time for the last %: "*string(round(ctime, 2))*"sec.")
            totaltime+=ctime
            println("Total elapsed time: "*string(round(totaltime, 2))*"sec.")
            println((minMatRho, minMatRho+length(vectRho)+maxLen-2))
            tic()
        end
        ###Measure of observables
        if tim[j]==tim2[cnt2]
            szObservable[cnt2]=real(trace(sz*reshape(buildReducedDensityMatrix(vectRho[1-minMatRho], 0), (2, 2))))
            sxObservable[cnt2]=real(trace(sx*reshape(buildReducedDensityMatrix(vectRho[1-minMatRho], 0), (2, 2))))
            cnt2+=1
        end
        if order==1
            computeDerivative!(vectDRho, vectRho, Hinside, leftApproximation, rightApproximation, tempMatrix)
            for l=1:length(vectDRho)
                @. vectRho[l]+= vectDRho[l]
            end
            minMatRho-=extendZone!(vectDRho, vectRho)
        elseif order==2
            computeDerivative!(vectDRho, vectRho, Hinside, leftApproximation, rightApproximation, tempMatrix)
                for l=1:length(vectDRho)
                    @. vectRho[l]+= 0.5*vectDRho[l]
                end
                for l=1:length(vectDRho)
                    @. vectDRhoAux[l].=vectRho[l]
                    @. vectDRhoAux[l]+=0.5*vectDRho[l]
                end
                computeDerivative!(vectDRho, vectDRhoAux, Hinside, leftApproximation, rightApproximation, tempMatrix)
                for l=1:length(vectRho)
                    @. vectRho[l]+= 0.5*vectDRho[l]
                end
                minMatRho-=extendZone!(vectDRho, vectDRhoAux, vectRho)
            #test[j]=minimum(minimum.(eigvals.(vectRho)))
         elseif order==4
            println("Not yet implemented")
        else
            println("Bug in order")
        end
    end
    toc()
    return tim2, szObservable, sxObservable#, test
end
