include("GeneralMethods.jl")
include("SuperOperator.jl")
include("Ising.jl")


###########################
###Core of the algorithm###
###########################
"""
    computeDerivative!(vectdRho, vectRho, Hinside, leftApprox, rightApprox, tempMatrix, Traces)

Compute the derivative of the densityMatrix
vectDRho stock the derivative
vectRho is the current vector of DM
Hinside is the Hamiltonian acting inside the DM
leftApprox correspond to the [TrL; TrR Tr L sz]^{-1}  [Tr ([sz_0 sz_1, .]); 0; 0]
rightApprox correspond to the [TrR; TrL Tr R sz]^{-1}  [Tr ([sl_0 sz_{l+1}}, .]); 0; 0]
tempMatrix is a simple temporary storage for some extremity matrix. Could be improved
Traces are the matrix of Trace operators
"""
function computeDerivative!(vectDRho, vectRho, Hinside, leftApprox, rightApprox, tempMatrix, Traces)
    insideEvolution!(vectDRho, vectRho, Hinside)
    outsideEvolution!(vectDRho, vectRho, leftApprox, rightApprox, tempMatrix, Traces)
end

"""
    insideEvolution!(vectDRho, vectRho, Hinside)

Compute the derivative of the densityMatrix according to Hinside
vectDRho stock the derivative
vectRho is the current vector of DM
Hinside is the Hamiltonian acting inside the DM
"""
function insideEvolution!(vectDRho, vectRho, Hinside)
    for j=1:length(vectRho)
        A_mul_B!(vectDRho[j], Hinside, vectRho[j])
    end
end

"""
    outsideEvolution!(vectdRho, vectRho, leftApprox, rightApprox, tempMatrix, Traces)

Increment the derivative of the densityMatrix by the link contribution
vectDRho stock the derivative
vectRho is the current vector of DM
leftApprox correspond to the [TrL; TrR Tr L sz]^{-1}  [Tr ([sz_0 sz_1, .]); 0; 0]
rightApprox correspond to the [TrR; TrL Tr R sz]^{-1}  [Tr ([sl_0 sz_{l+1}}, .]); 0; 0]
tempMatrix is a simple temporary storage for some extremity matrix. Could be improved
Traces are the matrix of Trace operators
"""
function outsideEvolution!(vectDRho, vectRho, leftApprox, rightApprox, tempMatrix, Traces)
    ###Auxiliary matrix to stock the boundaries  toDO=>removeThatStep
    leftFusionIdentity!(tempMatrix[1], reduceDensityMatrixRight(vectRho[1], Traces));
    rightFusionIdentity!(tempMatrix[2], reduceDensityMatrixLeft(vectRho[end], Traces));
    ###
    for j=1:length(vectDRho)-1
        vectDRho[j].=vectDRho[j]+rightApprox*vectRho[j+1]
    end
    vectDRho[end].=vectDRho[end]+rightApprox*tempMatrix[2]
    for j=2:length(vectDRho)
        vectDRho[j].=vectDRho[j]+leftApprox*vectRho[j-1]
    end
    vectDRho[1].=vectDRho[1]+leftApprox*tempMatrix[1]
end

"""
    extendZone!(vectDRho, vectRho, Traces)

Check if one needs to extend the considered range
vectDRho stock the derivative
vectRho is the current vector of DM
Traces are the matrix of Trace operators
"""
function extendZone!(vectDRho, vectRho, Traces)
    len=getLength(length(vectRho[1])); expanded=false
    ##Check if left is entangled
    mi=mutualInformation(vectRho[1], 0, 1, len, Traces)
    if mi>1e-7
        unshift!(vectRho, leftFusionIdentity(reduceDensityMatrixRight(vectRho[1], Traces)))
        unshift!(vectDRho, zeros(Complex128, length(vectDRho[1])))
        expanded=true
    end
    ##Same for right extremity
    mi=mutualInformation(vectRho[end], 0, len-1, len, Traces)
    if mi>1e-7
        push!(vectRho, rightFusionIdentity(reduceDensityMatrixLeft(vectRho[end], Traces)))
        push!(vectDRho, zeros(Complex128, length(vectDRho[1])))
    end
    if expanded
        return 1
    else
        return 0
    end
end

"""
    extendZone!(vectDRho, vectDRhoAux, vectRho, Traces)

Check if one needs to extend the considered range
vectDRho and vectDRhoAux stock the derivative
vectRho is the current vector of DM
Traces are the matrix of Trace operators
"""
function extendZone!(vectDRho, vectDRhoAux, vectRho, Traces)
    len=getLength(length(vectRho[1])); expanded=false
    ##Check if left is entangled
    mi=mutualInformation(vectRho[1], 0, 1, len, Traces)
    if mi>1e-7
        unshift!(vectRho, leftFusionIdentity(reduceDensityMatrixRight(vectRho[1], Traces)))
        unshift!(vectDRho, zeros(Complex128, length(vectDRho[1])))
        unshift!(vectDRhoAux, zeros(Complex128, length(vectDRhoAux[1])))
        expanded=true
    end
    ##Same for right extremity
    mi=mutualInformation(vectRho[end], 0, len-1, len, Traces)
    if mi>1e-7
        push!(vectRho, rightFusionIdentity(reduceDensityMatrixLeft(vectRho[end], Traces)))
        push!(vectDRho, zeros(Complex128, length(vectDRho[1])))
        push!(vectDRhoAux, zeros(Complex128, length(vectDRhoAux[1])))
    end
    if expanded
        return 1
    else
        return 0
    end
end

"""
    buildSpecialLinkProjector(maxLen, Hlink, Hcommuting)

Build the matrix operator corresponding to [TrL; TrR Tr L Hcommuting]^{-1}  [Tr ([sz_0 sz_1, .]); 0; 0]
and [TrR; TrL Tr R Hcommuting]^{-1}  [Tr ([sl_0 sz_{l+1}}, .]); 0; 0].
"""
function buildSpecialLinkProjector(maxLen, Hlink, Hcommuting)
    TrRight=buildTraceMatrixRight(maxLen-1);
    TrLeft=buildTraceMatrixLeft(maxLen-1);
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

"""
    buildSpecialLinkProjector_largeSize(maxLen, Hlink, Hcommuting)

Build the matrix operator corresponding to [TrL; TrR Tr L Hcommuting]^{-1}  [Tr ([sz_0 sz_1, .]); 0; 0]
and [TrR; TrL Tr R Hcommuting]^{-1}  [Tr ([sl_0 sz_{l+1}}, .]); 0; 0].
Useful for large matrices when the pseudoinverse cannot be taken
"""
function buildSpecialLinkProjector_largeSize(maxLen, Hlink, Hcommuting)
    TrRight=buildTraceMatrixRight(maxLen-1);
    TrLeft=buildTraceMatrixLeft(maxLen-1);
    ###For the left link
    leftCommuter=buildSuperOperatorLeft(kron(speye(2^(maxLen-1)), Hcommuting))
    MatJoin=sparse(vcat(TrRight, TrLeft, TrLeft*leftCommuter));
    tempLeftEvolutionOperator=speye(size(MatJoin)[1])[:, 1:2^(2*maxLen-2)]
    leftEvolutionOperator=spzeros(size(MatJoin)[2], 2^(2*maxLen-2))
    for j=1:2^(2*maxLen-2)
        leftEvolutionOperator[:, j]=triming.(MatJoin\full(tempLeftEvolutionOperator[:, j]))
    end
    leftEvolutionOperator=leftEvolutionOperator*TrLeft*Hlink[1]
    ###For the right link
    rightCommuter=buildSuperOperatorLeft(kron(Hcommuting, speye(2^(maxLen-1))))
    MatJoin=sparse(vcat(TrLeft, TrRight, TrRight*rightCommuter));
    tempRightEvolutionOperator=speye(size(MatJoin)[1])[:, 1:2^(2*maxLen-2)]
    rightEvolutionOperator=spzeros(size(MatJoin)[2], 2^(2*maxLen-2))
    for j=1:2^(2*maxLen-2)
        rightEvolutionOperator[:, j]=triming.(MatJoin\full(tempLeftEvolutionOperator[:, j]))
    end
    rightEvolutionOperator=rightEvolutionOperator*TrRight*Hlink[2]
    return leftEvolutionOperator, rightEvolutionOperator
end

###################
###Main function###
###################

###toDo look static arrays ?
function main(hx::Float64, hz::Float64, T::Float64; maxLen=3, dt=0.01, order=2, ini=1.0)
    #Define Pauli Matrices
    sx=[0 1; 1 0]; sz=-[1  0 ; 0 -1]; sy=[0 -1im; 1im 0]
    tim=0:dt:T
    #Define Hamiltonian
    Hinside=buildSuperCommutator(-1im*dt*buildInsideHamiltonian(hx, hz, maxLen));  #### Always multiplication on the right => slightly faster. Will have to recheck
    Hlink=buildSuperCommutator.(-1im*dt*buildLinkHamiltonian(hx, hz, maxLen));
    ###Setting up the global trace operators: note, the original one site reduced DM is marginally faster
    ###Choosing global matrixes seem to slow down code by ~1% but is much easier to read. More benchmark to be done later ?
    Traces=initializeTraceMatrices(maxLen);
    if maxLen<7
        leftApproximation, rightApproximation=buildSpecialLinkProjector(maxLen, Hlink, sparse(sz))
    else
        leftApproximation, rightApproximation=buildSpecialLinkProjector_largeSize(maxLen, Hlink, sparse(sz))
    end
    ###Initialization of the density operators.
    vectRho=map(x->reshape(x, length(x)), generateInitialState_z(maxLen, ini=ini));  minMatRho=-maxLen+1
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
    test=zeros(length(tim));
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
            szObservable[cnt2]=real(trace(sz*reshape(buildReducedDensityMatrix(vectRho[1-minMatRho], 0, Traces), (2, 2))))
            sxObservable[cnt2]=real(trace(sx*reshape(buildReducedDensityMatrix(vectRho[1-minMatRho], 0, Traces), (2, 2))))
            cnt2+=1
        end
        if order==1
            computeDerivative!(vectDRho, vectRho, Hinside, leftApproximation, rightApproximation, tempMatrix, Traces)
            for l=1:length(vectDRho)
                @. vectRho[l]+= vectDRho[l]
            end
            minMatRho-=extendZone!(vectDRho, vectRho, Traces)
        elseif order==2
            computeDerivative!(vectDRho, vectRho, Hinside, leftApproximation, rightApproximation, tempMatrix, Traces)
                for l=1:length(vectDRho)
                    @. vectRho[l]+= 0.5*vectDRho[l]
                end
                for l=1:length(vectDRho)
                    @. vectDRhoAux[l].=vectRho[l]
                    @. vectDRhoAux[l]+=0.5*vectDRho[l]
                end
                computeDerivative!(vectDRho, vectDRhoAux, Hinside, leftApproximation, rightApproximation, tempMatrix, Traces)
                for l=1:length(vectRho)
                    @. vectRho[l]+= 0.5*vectDRho[l]
                end
                minMatRho-=extendZone!(vectDRho, vectDRhoAux, vectRho, Traces)
            test[j]=minimum(minimum.(real.(eigvals.(map(x->reshape(x, (2^maxLen, 2^maxLen)), vectRho)))))
         elseif order==4
            println("Not yet implemented")
        else
            println("Bug in order")
        end
    end
    toc()
    return tim2, szObservable, sxObservable, test
end

#
# tim, sz, sx, test=main(0.25, -0.525, 100., maxLen=4, dt=0.0001, order=2, ini=0.6)
# tim2, sz2, sx2, test2=main(0.25, -0.525, 100., maxLen=5, dt=0.0001, order=2, ini=0.6)
# tim3, sz3, sx3, test3=main(0.25, -0.525, 100., maxLen=6, dt=0.0005, order=2, ini=0.6)
# tim4, sz4, sx4, test4=main(0.25, -0.525, 100., maxLen=7, dt=0.001, order=2, ini=0.6)
# timt, szt, sxt, testt=main(0.25, -0.525, 100., maxLen=4, dt=0.0001, order=2)
# timt2, szt2, sxt2, testt2=main(0.25, -0.525, 100., maxLen=5, dt=0.0001, order=2)
#
#
# figure()
# plot(log.(abs.(tim)), log.(abs.(sz/2*5)))
# plot(log.(abs.(tim2)), log.(abs.(sz2/2*5)))
# plot(log.(abs.(tim3)), log.(abs.(sz3/2*5)))
# plot(log.(abs.(tim4)), log.(abs.(sz4/2*5)))
# plot(log.(abs.(timt)), log.(abs.(szt/2)), ":")
# plot(log.(abs.(timt2)), log.(abs.(szt2/2)), ":")
# legend(["l=4", "l=5", "l=6", "l=7", "l=4 ori", "l=5 ori"])
#
# figure()
# plot(test)
# plot(test2)
# plot(test3)
# plot(test4)
# plot(testt)
# plot(testt2)
# legend(["l=4", "l=5", "l=6", "l=7", "l=4 ori", "l=5 ori"])
