########################################
###all hamiltonian-building functions###
########################################
"""
    buildLocalHamiltonian(hx, hz, maxLen)

Return an array of local hamiltonian :X_jX_{j+1} + 0.5*hx*(X[j]+X[j+1])+ 0.5*hz*(Z[j]+Z[j+1])
"""
function buildLocalHamiltonian(hx, hz, maxLen)
    dicHam=Dict()
    for site=1:maxLen-1
        dicHam[site]=spzeros(2^maxLen, 2^maxLen)
        for x=1:2^maxLen
            dig=2*digits(x-1, 2, maxLen)-1
            ###Kinetic terms and sz
            dicHam[site][x, x-dig[site]*2^(site-1)-dig[site+1]*2^(site)]+=0.25
            dicHam[site][x, x-dig[site]*2^(site-1)]+=hx/2/2
            dicHam[site][x, x]+=hz/2*dig[site]/2
        end
    end
    return dicHam
end

"""
    buildInsideHamiltonian(hx, hz, maxLen)

Return the Hamiltonian acting only on site  1:maxLen by dumping extremity links
"""
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


"""
    buildLinkHamiltonian(hx, hz, maxLen)

Return the array (left-right) of links: h_{1:2} and h_{maxLen-1:maxLen} for states defined on 1:maxLen
"""
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


"""
    generateInitialState_z(maxLen)

Project the spin 0 on +_z and return all DM of length maxLen containing it
"""
function generateInitialState_z(maxLen)
    listRho=Array{Array{Complex128}}(maxLen)
    localRho=[0.0+0.0im 0.0; 0.0 1.0]
    for l in 1:maxLen
        listRho[l]=kron(kron(eye(Complex128, 2^(l-1)),  localRho), eye(Complex128, 2^(maxLen-l)))/2^(maxLen-1)
    end
    return listRho
end

"""
    generateInitialState_x(maxLen)

Project the spin 0 on +_x and return all DM of length maxLen containing it
"""
function generateInitialState_x(maxLen)
    listRho=Array{Array{Complex128}}(maxLen)
    localRho=[0.5+0.0im 0.5; 0.5 0.5]
    for l in 1:maxLen
        listRho[l]=kron(kron(eye(Complex128, 2^(l-1)),  localRho), eye(Complex128, 2^(maxLen-l)))/2^(maxLen-1)
    end
    return listRho
end
