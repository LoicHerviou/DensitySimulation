include("GeneralMethods.jl")
include("SuperOperator.jl")
include("Ising.jl")


###########################
###Core of the algorithm###
###########################
"""
    joinDensityMatrix!(extendedRho, matRho, indexMin, indexMinTER, indexMax, indexMaxTER, Traces, TracesSmall)

Build rho_0:l from rho_0:l-1 and rho_1:l
extendedRho stocks the larger DMs
matRho are the current DMs
indexMin is the lefttmost site in matRho
indexMinTER is the leftmost site in matRho
indexMax is the rightmost site in matRho
indexMaxTER is the rightmost site in matRho
Traces are the trace matrix for extendedRho
TracesSmall are the trace matrix for matRho
"""
function joinDensityMatrix!(extendedRho, matRho, indexMin, indexMinTER, indexMax, indexMaxTER, Traces, TracesSmall, invMatJoin, pseudoIdentity)
    if indexMin==indexMinTER && indexMax==indexMaxTER
        if mod(length(matRho), 2)==1
            println("Bug1 in joinDensityMatrix 1")
        end
        for l in 1:2:length(matRho)-1
            joinDensityMatAux!(extendedRho[(l+1)÷2], matRho[l], matRho[l+1], Traces, TracesSmall, invMatJoin, pseudoIdentity)
        end
    elseif indexMin==indexMinTER && indexMax==indexMaxTER-1
        if mod(length(matRho), 2)==0
            println("Bug2 in joinDensityMatrix 2")
        end
        for l in 1:2:length(matRho)-1
            joinDensityMatAux!(extendedRho[(l+1)÷2], matRho[l], matRho[l+1], Traces, TracesSmall, invMatJoin, pseudoIdentity)
        end
        rightFusionIdentity!(extendedRho[end], matRho[end])
    elseif indexMin==indexMinTER+1 && indexMax==indexMaxTER
        if mod(length(matRho), 2)==0
            println("Bug3 in joinDensityMatrix 3")
        end
        for l in 2:2:length(matRho)-1
            joinDensityMatAux!(extendedRho[l÷2+1], matRho[l], matRho[l+1], Traces, TracesSmall, invMatJoin, pseudoIdentity)
        end
        leftFusionIdentity!(extendedRho[1], matRho[1])
    elseif indexMin==indexMinTER+1 && indexMax==indexMaxTER-1
        if mod(length(matRho), 2)==1
            println("Bug4 in joinDensityMatrix 4")
        end
        for l in 2:2:length(matRho)-1
            joinDensityMatAux!(extendedRho[l÷2+1], matRho[l], matRho[l+1], Traces, TracesSmall, invMatJoin, pseudoIdentity)
        end
        leftFusionIdentity!(extendedRho[1], matRho[1])
        rightFusionIdentity!(extendedRho[end], matRho[end])
    else
        println("Bug in joinDensityMatrix! 5")
    end
end

"""
    joinDensityMatAux!(rhoGuess, rhoLeft, rhoRight, Traces, TracesSmall)

Build rhoGuess from rhoLeft and rhoRight
rhoGuess is both the initial guess and the returned matrix
rhoLeft is the initial left matrix
rhoRight is the initial right matrix
Traces are the trace matrix for extendedRho
TracesSmall are the trace matrix for matRho
"""
function joinDensityMatAux!(rhoGuess, rhoLeft, rhoRight, Traces, TracesSmall, invMatJoin, pseudoIdentity)
    len=getLength(length(rhoLeft));
    # maxinconsistency=1e-8
    # consistency=max(norm(reduceDensityMatrixRight(rhoGuess, Traces)-rhoLeft, 2), norm(reduceDensityMatrixLeft(rhoGuess, Traces)-rhoRight, 2)) ###Is it needed ?
    # if consistency<maxinconsistency
    #     return 0
    # end
    ###Check if input matrices are positive -> ignored
    ###See which product ansatz is best
    # temprho1=buildReducedDensityMatrix(rhoLeft, 0, TracesSmall)
    # rho1=leftFusion(temprho1, reduceDensityMatrixRight(rhoRight, TracesSmall))
    # consistency1=norm(rho1-rhoLeft, 2)
    # if consistency1<maxinconsistency
    #     rhoGuess.=leftFusion(temprho1, rhoRight)
    #     return 1
    # end
    # temprho1=buildReducedDensityMatrix(rhoRight, len-1, TracesSmall)
    # rho1=rightFusion(reduceDensityMatrixLeft(rhoLeft, TracesSmall), temprho1)
    # consistency1=norm(rho1-rhoRight, 2)
    # if consistency1<maxinconsistency
    #     rhoGuess.=rightFusion(rhoLeft, temprho1)
    #     return 1
    # end
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
    massReduce!(matRho, timeEvolvedRho, indexmin, indexminTER, Traces)

Update matRho from the evolved larger DMs
matRho are the current DMs to be updated
timeEvolvedRho are the time evolved larger DMs
indexMin is the lefttmost site in matRho
indexMinTER is the leftmost site in matRho
Traces are the trace matrix for extendedRho
"""
function massReduce!(matRho, timeEvolvedRho, indexmin, indexminTER, Traces)
    if indexmin==indexminTER
        for l in 1:length(matRho)
            if mod(l, 2)==1
                reduceDensityMatrixRight!(matRho[l], timeEvolvedRho[(l+1)÷2], Traces)
            else
                reduceDensityMatrixLeft!(matRho[l], timeEvolvedRho[l÷2], Traces)
            end
            normalizeDM!(matRho[l])
        end
    elseif indexmin==indexminTER+1
        for l in 1:length(matRho)
            if mod(l, 2)==1
                reduceDensityMatrixLeft!(matRho[l], timeEvolvedRho[(l+1)÷2], Traces)
            else
                reduceDensityMatrixRight!(matRho[l], timeEvolvedRho[l÷2+1], Traces)
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

"""
    extendZone!(matRho, indexmin, indexmax, extendedCurrent, indicesCurrent, extendedOther, indicesOther, Traces, TracesSmall)

Check if one needs to extend the considered range
matRho are the current DMs
indexmin is the lefttmost site in matRho
indexmax is the rightmost site in matRho
extendedCurrent are the current extendedDM (odd or even)
indicesCurrent are the min and max index of extendedCurrent
extendedOther are the other extendedDM (odd or even)
indicesOther are the min and max index of extendedOther
Traces are the trace matrix for extendedRho
TracesSmall are the trace matrix for matRho
"""
function extendZone!(matRho, indexmin, indexmax, extendedCurrent, indicesCurrent, extendedOther, indicesOther, Traces, TracesSmall)
    len=getLength(length(matRho[1]));
    ##Check if left is entangled
    mini=indexmin; maxi=indexmax;
    mi=mutualInformation(matRho[1], 0, 1, len, TracesSmall)
    if mi>1e-7
        if indicesCurrent[1]==indexmin-1
            unshift!(matRho, reduceDensityMatrixRight(extendedCurrent[1], Traces))
        elseif indicesCurrent[1]==indexmin
            unshift!(matRho, leftFusionIdentity(reduceDensityMatrixRight(matRho[1], TracesSmall)))
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
            unshift!(extendedOther, leftFusionIdentity(leftFusionIdentity(reduceDensityMatrixRight(matRho[1], TracesSmall))))
            indicesOther[1]-=2
        end
    end
    ##Same for right extremity
    mi=mutualInformation(matRho[end], 0, len-1, len, TracesSmall)
    if mi>1e-7
        if indicesCurrent[2]>indexmax
            push!(matRho, reduceDensityMatrixLeft(extendedCurrent[end], Traces))
        elseif indicesCurrent[2]==indexmax
            push!(matRho, rightFusionIdentity(reduceDensityMatrixLeft(matRho[end], TracesSmall)))
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
            push!(extendedOther, rightFusionIdentity(rightFusionIdentity(reduceDensityMatrixLeft(matRho[end], TracesSmall))))
            indicesOther[2]+=2
        end
    end
    return mini, maxi
end




###################
###Main function###
###################
function main(hx::Float64, hz::Float64, T::Float64; maxLen=3, dt=0.0001)
    #Define Pauli Matrices
    sx=[0 1; 1 0]; sz=-[1  0 ; 0 -1]; sy=[0 -1im; 1im 0]
    tim=dt:dt:T
    #Define Hamiltonian
    dicLocalHamiltonianInside=buildLocalHamiltonian(hx, hz, maxLen)  #### Build local Hamiltonian i:i+1 for i=0:maxLen-1 for maxLen+1 sites matrices.
    dicLocalHamiltonian=buildLocalHamiltonian(hx, hz, maxLen+1)  #### Build local Hamiltonian i:i+1 for i=0:maxLen-1 for maxLen+1 sites matrices.
    ###Evolution operator for even and odd sites
    ###toDo: reexpress as evolution on vectors
    ULeft=expm(full(-1im*dt*sum([dicLocalHamiltonian[x] for x in 1:2:maxLen])))
    tULeft=ULeft'
    #StockedOperator
    extendedRho_o=Array{Array{Complex128, 1}}((maxLen+1)÷2); extendedRho_e=Array{Array{Complex128, 1}}((maxLen)÷2+1);
    for l in 1:length(extendedRho_o)
        extendedRho_o[l]=reshape(eye(2^(maxLen+1), 2^(maxLen+1))/2^(maxLen+1),  2^(maxLen*2+2))
    end
    for l in 1:length(extendedRho_e)
        extendedRho_e[l]=reshape(eye(2^(maxLen+1), 2^(maxLen+1))/2^(maxLen+1),  2^(maxLen*2+2))
    end
    ###Setting up the global trace operators: note, the original one site reduced DM is marginally faster
    ###Choosing global matrixes seem to slow down code by ~1% but is much easier to read. More benchmark to be done later ?
    Traces=initializeTraceMatrices(maxLen+1);
    TracesSmall=initializeTraceMatrices(maxLen);
    #if !isdefined(:MatJoin)
    TrRight=buildTraceMatrixRight(maxLen);
    Tr=buildTraceMatrix(maxLen);
    TrLeft=buildTraceMatrixLeft(maxLen);
    MatJoin=transpose(sparse(vcat(TrRight, TrLeft, Tr)));
    invMatJoin=sparse(triming.(pinv(full(MatJoin))));
    pseudoIdentity=sparse(triming.(MatJoin*invMatJoin));
    #end
    ###For memory efficience
    tempMatrix=reshape(similar(extendedRho_e[1]), (size(ULeft)[1],size(ULeft)[1]) )


    ###Observables###
    tim2=0:100*dt:T;
    szObservable=zeros((length(tim)+99)÷100+1);
    sxObservable=zeros((length(tim)+99)÷100+1);
    #initializeState
    matRho=map(x->reshape(x, length(x)), generateInitialState_z(maxLen));
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
            println((minMatRho, maxMatRho))
            tic()
        end
        if tim[j]==tim2[cnt2]
            szObservable[cnt2]=real(trace(sz*reshape(buildReducedDensityMatrix(matRho[1-minMatRho], 0, TracesSmall), (2, 2))))
            sxObservable[cnt2]=real(trace(sx*reshape(buildReducedDensityMatrix(matRho[1-minMatRho], 0, TracesSmall), (2, 2))))
            cnt2+=1
        end
        joinDensityMatrix!(extendedRho_o, matRho, minMatRho, indices_o[1], maxMatRho, indices_o[2], Traces, TracesSmall, invMatJoin, pseudoIdentity);   #Build the larger density Matrix
        evolveMatrix!(extendedRho_o, tempMatrix, ULeft, tULeft);   #Evolve the larger densityMatrix
        massReduce!(matRho, extendedRho_o, minMatRho, indices_o[1], Traces) #Compute the update densityMatrix
        minMatRho, maxMatRho=extendZone!(matRho, minMatRho, maxMatRho, extendedRho_o, indices_o, extendedRho_e, indices_e, Traces, TracesSmall)
        joinDensityMatrix!(extendedRho_e, matRho, minMatRho, indices_e[1], maxMatRho, indices_e[2], Traces, TracesSmall, invMatJoin, pseudoIdentity)   #Build the larger density Matrix
        evolveMatrix!(extendedRho_e, tempMatrix, ULeft, tULeft)
        massReduce!(matRho, extendedRho_e, minMatRho, indices_e[1], Traces)
        minMatRho, maxMatRho=extendZone!(matRho, minMatRho, maxMatRho, extendedRho_e, indices_e, extendedRho_o, indices_o, Traces, TracesSmall)
        ##Measure sz
    end
    toc()
    return tim2, szObservable, sxObservable
end


###ToDo: remplacer isposdef par issemiposdef
###ToDo: optimize matrix product in joinDensity ? Have an explicit function compiled probably.
