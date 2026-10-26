"""Regression checks for thermal handoffs and conserved film constituents."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from validation.multistage.functional_coating import pipeline
from validation.multistage.functional_coating.curing import physics_model as curing
from validation.multistage.src.stages import load
from validation.multistage.src.cli import main
from validation.multistage.tests.test_pipeline import nominal_line


class FilmConservationTests(unittest.TestCase):
    def inputs(self, **overrides):
        x = dict(oven_temperature_c=140., hold_time_min=45., nip_pressure_mpa=.3,
                 incoming_dry_thickness_um=100., incoming_residual_solvent_pct=30.,
                 incoming_internal_stress_mpa=.6, incoming_drying_defect_index=.1,
                 incoming_thickness_cv_fraction=.035)
        return {**x, **overrides}

    def test_mass_and_volume_close_for_wet_and_solvent_free_films(self):
        x = self.inputs(incoming_residual_solvent_pct=np.array([0., .5, 30., 70.]))
        r = curing.evaluate_model(**x)
        # Independently recover initial/final constituent masses from reported
        # thickness/concentration and the specified polymer densification law.
        w0 = x['incoming_residual_solvent_pct'] / 100.
        m0 = 100e-6 / ((1-w0)/1250. + w0/790.)
        contraction = .022*r['cure_fraction'] + .006*(1-np.exp(-.3/.18))
        polymer_density = 1250./(1-contraction)
        w1 = r['final_residual_solvent_pct']/100.
        m1 = r['final_thickness_um']*1e-6 / ((1-w1)/polymer_density+w1/790.)
        np.testing.assert_allclose(m1*(1-w1), m0*(1-w0), rtol=1e-12)
        self.assertTrue(np.all(m1*w1 <= m0*w0))
        self.assertEqual(float(w1[0]), 0.)
        self.assertTrue(np.all(r['final_thickness_um'] <= 100.))

    def test_concentration_uses_remaining_total_mass(self):
        # Constant 100 C removes quadrature/temperature approximations from
        # the analytic exponential solvent-mass decay reference.
        x = self.inputs(oven_temperature_c=100., incoming_film_temperature_c=100.)
        r = curing.evaluate_model(**x)
        remaining = np.exp(-.055 / (100./80.)**1.45 * 45.)
        expected = 100.*(.3*remaining)/(.7+.3*remaining)
        self.assertAlmostEqual(float(r['final_residual_solvent_pct']), expected, places=12)

    def test_line_passes_drying_temperature_exactly(self):
        with patch.object(curing, 'evaluate_model', wraps=curing.evaluate_model) as model:
            r = pipeline.evaluate_line(nominal_line())
        np.testing.assert_array_equal(model.call_args.kwargs['incoming_film_temperature_c'],
                                      r['drying']['terminal_film_temperature_c'])
        direct = load('curing').evaluate({
            **{k: nominal_line()[k] for k in curing.CONTROL_BOUNDS},
            'incoming_dry_thickness_um': r['drying']['dry_thickness_um'],
            'incoming_residual_solvent_pct': r['drying']['residual_solvent_pct'],
            'incoming_internal_stress_mpa': r['drying']['internal_stress_mpa'],
            'incoming_drying_defect_index': r['drying']['drying_defect_index'],
            'incoming_thickness_cv_fraction': r['drying']['dry_thickness_cv_fraction'],
            'incoming_film_temperature_c': r['drying']['terminal_film_temperature_c'],
        })
        for key in direct:
            np.testing.assert_array_equal(direct[key], r['curing'][key])

    def test_temperature_broadcast_cooling_and_legacy_default(self):
        x = self.inputs()
        default = load('curing').evaluate(x)
        explicit = load('curing').evaluate({**x, 'incoming_film_temperature_c':25.})
        for key in default:
            np.testing.assert_array_equal(default[key], explicit[key])
        r = curing.evaluate_model(**self.inputs(oven_temperature_c=90.,
            incoming_film_temperature_c=np.array([[25.],[150.]]),
            hold_time_min=np.array([5.,30.])))
        t = r['terminal_film_temperature_c']
        self.assertEqual(t.shape, (2,2))
        self.assertTrue(np.all((t[0]>25)&(t[0]<90)))
        self.assertTrue(np.all((t[1]>90)&(t[1]<150)))
        for bad in (np.nan, np.inf, 24., 181.):
            with self.assertRaises(ValueError):
                curing.evaluate_model(**self.inputs(incoming_film_temperature_c=bad))

    def continuous_inputs(self):
        return {k:v for k,v in nominal_line().items()
                if k not in ('residence_time_min','hold_time_min')}

    def test_fixed_oven_lengths_couple_speed_dwell_and_throughput(self):
        x = self.continuous_inputs()
        x['line_speed_m_min'] = np.array([18.,24.])
        r = pipeline.evaluate_continuous_line(x, drying_length_m=234.,
                                              curing_length_m=1080., web_width_m=2.)
        np.testing.assert_allclose(r['inputs']['residence_time_min'],[13.,9.75])
        np.testing.assert_allclose(r['inputs']['hold_time_min'],[60.,45.])
        np.testing.assert_allclose(r['final']['line_throughput_m2_h'],[2160.,2880.])
        direct = pipeline.evaluate_line(r['inputs'])
        for key in direct['curing']:
            np.testing.assert_array_equal(direct['curing'][key], r['curing'][key])

    def test_invalid_geometry_conflicting_times_and_short_ovens_rejected(self):
        for bad in (0.,-1.,np.nan,np.inf,[234.]):
            with self.assertRaises(ValueError):
                pipeline.evaluate_continuous_line(self.continuous_inputs(),
                    drying_length_m=bad,curing_length_m=1080.)
        with self.assertRaises(ValueError):
            pipeline.evaluate_continuous_line(nominal_line(),
                drying_length_m=234.,curing_length_m=1080.)
        with self.assertRaises(ValueError):
            pipeline.evaluate_continuous_line(self.continuous_inputs(),
                drying_length_m=1.,curing_length_m=1080.)

    def test_continuous_cli_and_gui_temperature_field(self):
        args=['evaluate-line','--drying-length-m','234','--curing-length-m','1080',
              '--set',*[f'{k}={v}' for k,v in self.continuous_inputs().items()]]
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(args),0)
        self.assertEqual(json.loads(output.getvalue())['inputs']['hold_time_min'],60.)
        from validation.konogui import _multistage_specs
        spec=next(s for s in _multistage_specs() if s.id=='curing')
        field=next(f for f in spec.inputs if f.name=='incoming_film_temperature_c')
        self.assertEqual(field.initial_value(),25.)


if __name__ == '__main__':
    unittest.main()
