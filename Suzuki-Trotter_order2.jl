include("GeneralMethods.jl")
include("SuperOperator.jl")
include("Ising.jl")

#using SymPy

###########################
###Core of the algorithm###
###########################
"""
    joinDensityMatrix!(extendedRho, matRho, TracesSmall, invMatJoin, pseudoIdentity)

Build rho_0:l+1 from rho_0:l-1, rho_1:l and rho_2:l+1
extendedRho stocks the larger DMs
matRho are the current DMs
TracesSmall are the trace matrix for matRho
invMatJoin is the pseudo-inverse to find the marginal
pseudoIdentity is used to minimize the distance with the previous matrix
"""
function joinDensityMatrix!(extendedRho, matRho, TracesSmall, invMatJoin, pseudoIdentity)
    for l in 2:length(matRho)-1
        joinDensityMatAux!(extendedRho[l],  matRho[l-1], matRho[l], matRho[l+1], invMatJoin, pseudoIdentity)
    end
    joinDensityMatAux!(extendedRho[1],  leftFusionIdentity(reduceDensityMatrixRight(matRho[1], TracesSmall)), matRho[1], matRho[2], invMatJoin, pseudoIdentity)
    joinDensityMatAux!(extendedRho[end],  matRho[end-1], matRho[end], rightFusionIdentity(reduceDensityMatrixLeft(matRho[end], TracesSmall)), invMatJoin, pseudoIdentity)
end

"""
    joinDensityMatAux!(rhoGuess, rhoLeft, rhoMiddle,  rhoRight, invMatJoin, pseudoIdentity)

Build rhoGuess from rhoLeft, rhoMiddle and rhoRight
rhoGuess is both the initial guess and the returned matrix
rhoLeft is the initial left matrix
rhoMiddle is the initial middle matrix, that we are approximating
rhoRight is the initial right matrix
invMatJoin is the pseudo-inverse to find the marginal
pseudoIdentity is used to minimize the distance with the previous matrix
"""
function joinDensityMatAux!(rhoGuess, rhoLeft, rhoMiddle,  rhoRight, invMatJoin, pseudoIdentity)
    len=getLength(length(rhoGuess));
    ###Fast method
    temp=transpose(At_mul_B(vcat(rhoLeft, rhoMiddle, rhoRight), invMatJoin))
    temp-=transpose(At_mul_B(rhoGuess, pseudoIdentity))
    temp/=2
    tempconj=reshape(transpose(conj.(reshape(temp, (2^(len), 2^(len))))), length(temp))  ###A ameliorer
    for j=1:length(rhoGuess)
        rhoGuess[j]+=temp[j]+tempconj[j]
    end
    return 0
end

"""
    evolveMatrix!(extendedRho, ULeft, tULeft)

Let the extended matrix evolve under Uleft and its transconjugate. ToDO: improve
"""
function evolveMatrix!(extendedRho,temp,  ULeft, tULeft)
    lenmat=size(ULeft)[1];
    for l in 1:length(extendedRho)
        A_mul_B!(temp,ULeft, reshape(view(extendedRho[l], :), (lenmat, lenmat)))
        A_mul_B!(reshape(view(extendedRho[l], :), (lenmat, lenmat)), temp, tULeft)
    end
end


"""
    massReduce!(matRho, timeEvolvedRho, Traces)

Update matRho from the evolved larger DMs
matRho are the current DMs to be updated
timeEvolvedRho are the time evolved larger DMs
Traces are the trace matrix for extendedRho
"""
function massReduce!(matRho, timeEvolvedRho, Traces)
    len=getLength(length(matRho[1]));
    for l=1:length(matRho)
        matRho[l]=buildReducedDensityMatrix(timeEvolvedRho[l], 1, len, Traces)
        normalizeDM!(matRho[l])
    end
end

"""
    extendZone!(matRho,  indexmin, extendedRho, Traces, TracesSmall)

Check if one needs to extend the considered range
matRho are the current DMs
indexmin is the lefttmost site in matRho
extendedRho are the current extendedDM (odd or even)
Traces are the trace matrix for extendedRho
TracesSmall are the trace matrix for matRho
"""
function extendZone!(matRho,  indexmin, extendedRho, Traces, TracesSmall)
    len=getLength(length(matRho[1]));
    ##Check if left is entangled
    mini=indexmin;
    mi=mutualInformation(matRho[1], 0, 1, len, TracesSmall)
    if mi>1e-7
        unshift!(matRho, buildReducedDensityMatrix(extendedRho[1], 0, len, Traces))
        unshift!(extendedRho, leftFusionIdentity(reduceDensityMatrixRight(extendedRho[1], Traces)))
        mini-=1
    end
    ##Same for right extremity
    mi=mutualInformation(matRho[end], 0, len-1, len, TracesSmall)
    if mi>1e-7
        push!(matRho, buildReducedDensityMatrix(extendedRho[end], 2, len, Traces))
        push!(extendedRho, rightFusionIdentity(reduceDensityMatrixLeft(extendedRho[end], Traces)))
    end
    return mini
end

###################
###Main function###
###################
function main(hx::Float64, hz::Float64, T::Float64; maxLen=3, dt=0.0001)
    #Define Pauli Matrices
    sx=[0 1; 1 0]; sz=-[1  0 ; 0 -1]; sy=[0 -1im; 1im 0]
    tim=dt:dt:T
    #Define Hamiltonian
    localHamiltonianSingleSite=buildSingleSiteHamiltonian(hx, hz, maxLen+2)  #### Build local Hamiltonian i:i+1 for i=0:maxLen-1 for maxLen+1 sites matrices.
    localHamiltonianMultiSite=buildMultiSiteHamiltonian(hx, hz, maxLen+2)  #### Build local Hamiltonian i:i+1 for i=0:maxLen-1 for maxLen+1 sites matrices.
    ###Evolution operator
    ULeft=expm(full(-1im*dt/2*localHamiltonianSingleSite))*expm(full(-1im*dt*localHamiltonianMultiSite))*expm(full(-1im*dt/2*localHamiltonianSingleSite))
    tULeft=ULeft'
    #StockedOperator
    extendedRho=Array{Array{Complex128, 1}}((maxLen));
    for l in 1:length(extendedRho)
        extendedRho[l]=reshape(eye(2^(maxLen+2), 2^(maxLen+2))/2^(maxLen+2),  2^(maxLen*2+4))
    end
    tempMatrix=reshape(similar(extendedRho[1]), (size(ULeft)[1],size(ULeft)[1]))
    ###Setting up the global trace operators
    Traces=initializeTraceMatrices(maxLen+2);
    TracesSmall=initializeTraceMatrices(maxLen);
    MatJoin=transpose(sparse(vcat(Traces[1, 3]', Traces[2,2]', Traces[3,1]')));
    invMatJoin=sparse(triming.(pinv(full(MatJoin))));
    pseudoIdentity=sparse(triming.(MatJoin*invMatJoin));
    ###Observables###
    if length(tim)<=1000
        tim2=0:10*dt:T;
    else
        tim2=0:100*dt:T;
    end
    szObservable=zeros((length(tim)+99)÷100+1);
    sxObservable=zeros((length(tim)+99)÷100+1);
    #initializeState
    matRho=map(x->reshape(x, length(x)), generateInitialState_z(maxLen));
    minMatRho=-maxLen+1;
    #timeEvolution
    tic()
    cnt=0; totaltime=0; cnt2=2;
    szObservable[1]=real(trace(sz*reshape(buildReducedDensityMatrix(matRho[1-minMatRho], 0, TracesSmall), (2, 2))))
    sxObservable[1]=real(trace(sx*reshape(buildReducedDensityMatrix(matRho[1-minMatRho], 0, TracesSmall), (2, 2))))
    for j in 1:length(tim)
        ####Convention: apply Odd then Even
        if mod(j, (length(tim)+99)÷100)==0
            cnt+=1
            ctime=toq()
            println(string(cnt)*"% of simulation done. Time for the last %: "*string(round(ctime, 2))*"sec.")
            totaltime+=ctime
            println("Total elapsed time: "*string(round(totaltime, 2))*"sec.")
            println((minMatRho, minMatRho+length(matRho)+maxLen-2))
            tic()
        end
        if tim[j]==tim2[cnt2]
            szObservable[cnt2]=real(trace(sz*reshape(buildReducedDensityMatrix(matRho[1-minMatRho], 0, TracesSmall), (2, 2))))
            sxObservable[cnt2]=real(trace(sx*reshape(buildReducedDensityMatrix(matRho[1-minMatRho], 0, TracesSmall), (2, 2))))
            cnt2+=1
        end
        joinDensityMatrix!(extendedRho, matRho, TracesSmall, invMatJoin, pseudoIdentity);   #Build the larger density Matrix
        evolveMatrix!(extendedRho, tempMatrix, ULeft, tULeft);   #Evolve the larger densityMatrix
        massReduce!(matRho, extendedRho, Traces) #Compute the update densityMatrix
        minMatRho=extendZone!(matRho, minMatRho, extendedRho, Traces, TracesSmall)
        ##Measure sz
    end
    toc()
    return tim2, szObservable, sxObservable
end


###ToDo: remplacer isposdef par issemiposdef
###ToDo: optimize matrix product in joinDensity ? Have an explicit function compiled probably.
