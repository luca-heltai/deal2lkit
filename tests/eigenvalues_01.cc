/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2014 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 
 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */



#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/base/index_set.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/slepc_spectral_transformation.h>


#include <deal.II/base/mpi.h>

#include <fstream>
#include <iostream>

#include "parsed_data_out.h"
#include "error_handler.h"
#include "utilities.h"

using namespace dealii;



template<int dim>
class EigenProblem : public ParameterAcceptor
{
public:
    
    EigenProblem ();
    
    virtual void declare_parameters(ParameterHandler &prm);
    
    void run ();
    
    
private:
    void make_grid_fe ();
    void setup_system ();
    void assemble_system ();
    void solve ();
    void output_results ();
    void process_solution (const unsigned int cycle);
    void refine_grid ();
    
    shared_ptr<Triangulation<dim> >    triangulation;
    shared_ptr<FiniteElement<dim,dim> >             fe;
    shared_ptr<DoFHandler<dim> >       dof_handler;
    
    ConstraintMatrix     constraints;
    PETScWrappers::SparseMatrix             stiffness_matrix, mass_matrix;
    std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
    std::vector<double>                     eigenvalues;
    
    ParsedDataOut<dim,dim> data_out;
    
    unsigned int n_cycles;
    unsigned int initial_refinement;
    unsigned int n_eigenvalues;
    unsigned int degree;
    std::vector<unsigned int> eigenvalue_refinement_indices;
    
    ErrorHandler<1> eh;
};


template <int dim>
EigenProblem<dim>::EigenProblem ()
:
ParameterAcceptor("Global parameters"),
data_out("Data out", "vtu")
{}


template <int dim>
void EigenProblem<dim>::declare_parameters(ParameterHandler &prm)
{
    
    add_parameter(prm, &initial_refinement, "Initial global refinement", "4",
                  Patterns::Integer(0));
    
    add_parameter(prm, &n_cycles, "Total number of cycles", "5",
                  Patterns::Integer(0));
    
    
    add_parameter(prm, &n_eigenvalues, "Number of eigenvalues to compute", "5",
                  Patterns::Integer(0));

    
    add_parameter(prm, &degree, "Degree of the pressure space", "0",
                  Patterns::Integer(0));
    

    add_parameter(prm, &eigenvalue_refinement_indices, "Eigenvector indices for a posteriori estimate", "",
                  Patterns::List(Patterns::Integer(0),0));

}

template <int dim>
void EigenProblem<dim>::make_grid_fe ()
{
    triangulation = SP(new Triangulation<dim>());
    GridGenerator::hyper_cube (*triangulation, 0, numbers::PI);

    
    triangulation->refine_global(initial_refinement);
    
    dof_handler = SP(new DoFHandler<dim>(*triangulation));
    
    std::cout << "Number of active cells: "
    << triangulation->n_active_cells()
    << std::endl;
    
    fe=SP(new FESystem<dim>(FE_RaviartThomas<dim>(degree), 1, FE_DGQ<dim>(degree), 1));
    
}



template <int dim>
void EigenProblem<dim>::setup_system ()
{
    dof_handler->distribute_dofs (*fe);
    DoFRenumbering::component_wise (*dof_handler);
    
    std::vector<types::global_dof_index> dofs_per_component (dim+1);
    DoFTools::count_dofs_per_component (*dof_handler, dofs_per_component);
    const unsigned int n_u = dofs_per_component[0],
    n_p = dofs_per_component[dim];
    std::cout << "Number of active cells: "
    << triangulation->n_active_cells()
    << std::endl
    << "Total number of cells: "
    << triangulation->n_cells()
    << std::endl
    << "Number of degrees of freedom: "
    << dof_handler->n_dofs()
    << " (" << n_u << '+' << n_p << ')'
    << std::endl;
    
    constraints.clear ();
    
    DoFTools::make_hanging_node_constraints (*dof_handler, constraints);
    
    constraints.close ();
    
    stiffness_matrix.reinit (dof_handler->n_dofs(),
                             dof_handler->n_dofs(),
                             dof_handler->max_couplings_between_dofs());
    mass_matrix.reinit (dof_handler->n_dofs(),
                        dof_handler->n_dofs(),
                        dof_handler->max_couplings_between_dofs());
    
    IndexSet eigenfunction_index_set = dof_handler->locally_owned_dofs ();
    eigenfunctions
    .resize (n_eigenvalues);
    for (unsigned int i=0; i<n_eigenvalues; ++i)
        eigenfunctions[i].reinit (eigenfunction_index_set, MPI_COMM_WORLD);
    eigenvalues.resize (eigenfunctions.size ());
}


template <int dim>
void EigenProblem<dim>::assemble_system ()
{
    QGauss<dim>   quadrature_formula(degree+2);
    QGauss<dim-1> face_quadrature_formula(degree+2);
    FEValues<dim> fe_values (*fe, quadrature_formula,
                             update_values    | update_gradients |
                             update_quadrature_points  | update_JxW_values);
    FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
                                      update_values    | update_normal_vectors |
                                      update_quadrature_points  | update_JxW_values);
    const unsigned int   dofs_per_cell   = fe->dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();
    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>   local_mass_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);
    
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler->begin_active(),
    endc = dof_handler->end();
    for (; cell!=endc; ++cell)
    {
        fe_values.reinit (cell);
        local_matrix = 0;
        local_mass_matrix = 0;
 
        for (unsigned int q=0; q<n_q_points; ++q)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                const Tensor<1,dim> phi_i_u     = fe_values[velocities].value (i, q);
                const double        div_phi_i_u = fe_values[velocities].divergence (i, q);
                const double        phi_i_p     = fe_values[pressure].value (i, q);
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    const Tensor<1,dim> phi_j_u     = fe_values[velocities].value (j, q);
                    const double        div_phi_j_u = fe_values[velocities].divergence (j, q);
                    const double        phi_j_p     = fe_values[pressure].value (j, q);
                    local_matrix(i,j) += (phi_i_u * phi_j_u
                                          +div_phi_i_u * phi_j_p
                                          +phi_i_p * div_phi_j_u)
                    * fe_values.JxW(q);
                    
                    local_mass_matrix(i,j) += (-phi_i_p*phi_j_p*fe_values.JxW(q));
                }
            }
        
        cell->get_dof_indices (local_dof_indices);
        
        constraints.distribute_local_to_global(local_matrix,
                                               local_dof_indices,
                                               stiffness_matrix);

        constraints.distribute_local_to_global(local_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
    }
    stiffness_matrix.compress (VectorOperation::add);
    mass_matrix.compress (VectorOperation::add);
    
    double min_spurious_eigenvalue = std::numeric_limits<double>::max(),
    max_spurious_eigenvalue = -std::numeric_limits<double>::max();
    for (unsigned int i = 0; i < dof_handler->n_dofs(); ++i)
        if (constraints.is_constrained(i))
        {
            stiffness_matrix.set(i, i, 1.234e5);
            mass_matrix.set(i, i, 1);
            const double ev = stiffness_matrix(i,i)/mass_matrix(i,i);
            min_spurious_eigenvalue = std::min (min_spurious_eigenvalue, ev);
            max_spurious_eigenvalue = std::max (max_spurious_eigenvalue, ev);
        }
    
    std::cout << "   Spurious eigenvalues are all in the interval "
    << "[" << min_spurious_eigenvalue << "," << max_spurious_eigenvalue << "]"
    << std::endl;

}


template <int dim>
void EigenProblem<dim>::solve ()
{
    SolverControl solver_control (dof_handler->n_dofs(), 1e-9);
    SLEPcWrappers::SolverArnoldi eigensolver (solver_control);
    SLEPcWrappers::TransformationCayley cayley(1.0, 1.0);
    
    eigensolver.set_which_eigenpairs (EPS_SMALLEST_MAGNITUDE);
    eigensolver.set_problem_type (EPS_GHIEP);
    // eigensolver.set_transformation(cayley);
    eigensolver.solve (stiffness_matrix, mass_matrix,
                       eigenvalues, eigenfunctions,
                       eigenfunctions.size());
    
    for (unsigned int i=0; i<eigenfunctions.size(); ++i) {
        constraints.distribute (eigenfunctions[i]);
        eigenfunctions[i] /= eigenfunctions[i].linfty_norm ();
    }
}


template <int dim>
void EigenProblem<dim>::output_results ()
{
    data_out.prepare_data_output(*dof_handler);
    for (unsigned int i=0; i<eigenfunctions.size(); ++i) {
        std::string toi=Utilities::int_to_string(i);
        data_out.add_data_vector (eigenfunctions[i], "u"+toi+",u"+toi+",p"+toi);
    }
    data_out.write_data_and_clear();
}

template <int dim>
void EigenProblem<dim>::process_solution (const unsigned int cycle)
{
    std::vector<double> exact_eigenvalues;
    exact_eigenvalues.push_back(2);
    exact_eigenvalues.push_back(5);
    exact_eigenvalues.push_back(5);
    exact_eigenvalues.push_back(8);
    exact_eigenvalues.push_back(10);
    exact_eigenvalues.push_back(10);
    exact_eigenvalues.push_back(13);
    exact_eigenvalues.push_back(13);
    exact_eigenvalues.push_back(16);
    
    for(unsigned int i=0; i<eigenvalues.size(); ++i) {
        eh.custom_error([this, i, exact_eigenvalues] (const unsigned int) {
            return eigenvalues[i]-exact_eigenvalues[i];
        }, *dof_handler, "l"+Utilities::int_to_string(i), i==0);
    }
}

template <int dim>
void EigenProblem<dim>::refine_grid ()
{
    if (eigenvalue_refinement_indices.size() == 0)
    {
        triangulation->refine_global (1);
    }
    else
    {
        Vector<float> estimated_error_per_cell (triangulation->n_active_cells());
        for(unsigned int i=0; i<eigenvalue_refinement_indices.size(); ++i) {
            KellyErrorEstimator<dim>::estimate (*dof_handler,
                                                QGauss<dim-1>(3),
                                                typename FunctionMap<dim>::type(),
                                                eigenfunctions[i],
                                                estimated_error_per_cell);
        }
        
        GridRefinement::refine_and_coarsen_fixed_number (*triangulation,
                                                         estimated_error_per_cell,
                                                         0.3, 0.0);
        
        triangulation->execute_coarsening_and_refinement ();
    }
}

template <int dim>
void EigenProblem<dim>::run ()
{
    make_grid_fe ();
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
    {
        std::cout<<"cycle : "<<cycle+1<<std::endl;
        setup_system ();
        assemble_system ();
        solve ();
        process_solution (cycle);
        if (cycle < n_cycles-1)
            refine_grid ();
        else
            output_results ();
    }
    eh.output_table(std::cout);
    
}



int main (int argc, char *argv[])
{
    
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    
    EigenProblem<2> laplace_problem_2;
    ParameterAcceptor::initialize("parameters_ser.prm", "used_parameters_ser.prm");
    laplace_problem_2.run ();
    
    return 0;
}
