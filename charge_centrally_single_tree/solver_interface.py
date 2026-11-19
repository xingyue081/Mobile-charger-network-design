"""只读统计产物适配层；不标定、不建模、不隐式调用求解器。"""
from dataclasses import dataclass
from pathlib import Path
from itertools import product
import hashlib
import json
import math
import numpy as np
import pandas as pd

EVENTS = ('workday', 'weekend', 'holiday')
SCALARS = ('mobile_unit_cost', 'fixed_unit_cost', 'rho', 'shortage_penalty',
           'fixed_capacity_per_unit', 'mobile_capacity_per_unit', 'transport_cost_per_distance')
MAPPINGS = ('hub_open_cost', 'mobile_fleet_ub', 'fixed_count_ub', 'deployment_ub', 'distance')
REQUIRED_BUSINESS = ('J',) + SCALARS + MAPPINGS


class InterfaceValidationError(ValueError):
    """输入损坏或不一致；禁止生成可调用的 solver 输入。"""


class SolverNotReadyError(RuntimeError):
    pass


@dataclass
class SolverInputs:
    data: dict
    params: dict
    metadata: dict

    def require_ready(self):
        """调用者必须显式通过此门禁；不会启动优化。"""
        if self.metadata['status'] != 'READY_FOR_REAL_DATA_SMOKE_TEST':
            raise SolverNotReadyError(json.dumps(self.metadata['blockers'], ensure_ascii=False))
        return self.data, self.params


def _check(ok, message):
    if not ok:
        raise InterfaceValidationError(message)


def _close(a, b, label, atol=1e-12):
    _check(np.allclose(a, b, rtol=0, atol=atol), label)


def _number(value, name, positive=False):
    _check(not isinstance(value, bool), name + ': boolean is not numeric input')
    v = float(value)
    _check(math.isfinite(v) and (v > 0 if positive else v >= 0), name + ': invalid value')
    return v


def build_solver_inputs(module2_dir, module3_dir, business_input=None, c_epsilon=None,
                        *, solver_options=None):
    """返回 SolverInputs(data, params, metadata)。

    business_input 是显式冻结的 mapping，使用现有 solver 字段名；不广播、无默认业务值。
    缺失业务值保留为报告 blocker，require_ready() 抛错；损坏产物立即抛错。
    solver_options 仅允许 separation_tolerance / gurobi / separation_gurobi /
    optimized_separation_gurobi。后者仅传给已验证的 optimized oracle
    的搜索参数命名空间，绝不传给 Restricted Master。
    若 module3 同级目录存在已验证的 wasserstein_radius_calibration_outputs，
    自动加载其最终 epsilon_l；否则 c_epsilon=None 时不含半径占位值。
    """
    root2, root3 = Path(module2_dir), Path(module3_dir)
    hashes = {}

    def read(path, kind='json'):
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        if kind == 'csv':
            return pd.read_csv(path, float_precision='round_trip')
        return json.loads(path.read_text(encoding='utf-8'))

    m2 = read(root2 / 'manifest.json')
    m3 = read(root3 / 'run_manifest.json')
    _check(m3['status'] == 'SUCCESS', 'module3 manifest is not SUCCESS')
    # 文件列表来自已审计的 manifest；未知附加文件不影响当前 schema。
    needed = ('statistical_parameters.npz', 'support_definition.json', 'support_membership.csv',
              'event_index_sets.csv', 'eventwise_support.csv', 'eventwise_empirical_samples.csv',
              'covariance_diagnostics.json', 'epsilon_status.json')
    _check(set(needed).issubset(m3['output_files']), 'module3 manifest missing required artifacts')
    for filename in m3['output_files']:
        _check(Path(filename).name == filename and (root3 / filename).is_file(),
               'invalid or missing manifest output: ' + filename)
        path = root3 / filename
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    order = read(root2 / 'taz_vector_order.csv', 'csv')
    I = tuple(int(i) for i in order['taz_id'])
    d = len(I)
    _check(d > 0 and len(set(I)) == d and list(order['taz_id']) == list(I), 'malformed TAZ IDs')
    _check(list(order['vector_position']) == list(range(1, d+1)), 'TAZ vector_position ordering')
    _check(list(I) == m2['taz_vector_order'] == m3['taz_ids'], 'TAZ ordering differs from manifest')
    samples = read(root2 / 'joint_historical_samples.csv', 'csv')
    _check(hashes[str((root2 / 'joint_historical_samples.csv').resolve())] == m3['input_sha256'],
           'module3 input_sha256 does not match module2 samples')
    cols = ['D_taz_' + str(i) for i in I]
    _check(list(samples.columns) == ['n', 'date', 's_n'] + cols, 'demand TAZ column ordering')
    N = len(samples)
    _check(N == m2['N'] == m3['N'] and d == m3['dimension'], 'manifest count/dimension mismatch')
    _check(list(samples.n) == list(range(1, N+1)), 'global IDs must be chronological 1..N')
    dates = pd.to_datetime(samples.date, format='%Y-%m-%d', errors='raise')
    _check(dates.is_unique and dates.is_monotonic_increasing, 'dates duplicated or reordered')
    interval = m2['training_interval']
    _check(((dates >= interval['start_inclusive']) & (dates < interval['end_exclusive'])).all(),
           'sample outside training interval')
    _check(tuple(m3['date_types']) == EVENTS and set(samples.s_n) == set(EVENTS), 'date types mismatch')
    membership2 = read(root2 / 'sample_membership.csv', 'csv')
    _check(membership2.equals(samples[['n', 'date', 's_n']]), 'module2 sample membership mismatch')
    D = samples[cols].to_numpy(dtype=float)
    _check(D.shape == (N, d) and np.isfinite(D).all() and (D >= 0).all(), 'demand shape/values')
    npz_path = root3 / 'statistical_parameters.npz'
    with np.load(npz_path, allow_pickle=False) as obj:
        mu, sigma_hat, sigma, ids = (obj[k].copy() for k in ('mu_hat', 'Sigma_hat', 'Sigma_tilde', 'taz_ids'))
    _check(ids.shape == (d,) and ids.tolist() == list(I), 'covariance TAZ ordering')
    _check(mu.shape == (d,) and sigma_hat.shape == sigma.shape == (d, d), 'statistical shape mismatch')
    _check(all(np.isfinite(a).all() for a in (mu, sigma_hat, sigma)), 'nonfinite statistical object')
    _close(sigma, sigma.T, 'Sigma_tilde not symmetric')
    support = read(root3 / 'support_definition.json')
    diag = read(root3 / 'covariance_diagnostics.json')
    _check(support['pooled'] is True, 'support must be global pooled')
    _check(diag['matrix_dimension'] == [d, d], 'diagnostic dimension mismatch')
    if support['Sigma_tilde_provenance'] == 'Sigma_tilde=Sigma_hat; no regularization applied':
        _check(np.array_equal(sigma, sigma_hat), 'declared unregularized Sigma_tilde != Sigma_hat')
    r = _number(support['r'], 'r')
    r2 = _number(support['r_squared'], 'r_squared')
    _close(r*r, r2, 'r squared mismatch')
    try:
        L = np.linalg.cholesky(sigma)  # 分解已有矩阵，不估计、不正则化。
    except np.linalg.LinAlgError as exc:
        raise InterfaceValidationError('Sigma_tilde not positive definite; solver incompatible') from exc
    membership = read(root3 / 'support_membership.csv', 'csv')
    _check(membership[['n', 'date', 'date_type']].rename(columns={'date_type': 's_n'}).equals(
        samples[['n', 'date', 's_n']]), 'support membership sample identity mismatch')
    _check(membership.inside_omega.dtype == bool, 'inside_omega must contain booleans')
    distances = membership.mahalanobis_sq.to_numpy(float)
    _check(np.isfinite(distances).all() and (distances >= 0).all(), 'invalid stored Mahalanobis distances')
    _check(np.array_equal(membership.inside_omega, distances <= r2), 'support membership contradicts r_squared')
    # 只对已冻结 support 作代入核验，不重估参数或替换 membership。
    whitened = np.linalg.solve(L, (D-mu).T)
    _close((whitened*whitened).sum(axis=0), distances, 'stored support distances inconsistent', atol=1e-10)
    tables = [read(root2 / 'date_type_statistics.csv', 'csv'),
              pd.DataFrame(m2['date_types']), read(root3 / 'event_index_sets.csv', 'csv'),
              read(root3 / 'eventwise_support.csv', 'csv')]
    for table in tables:
        _check(list(table.date_type) == list(EVENTS), 'event table order/duplicates')
    stats, _, indices, event_support = [t.set_index('date_type') for t in tables]
    empirical = read(root3 / 'eventwise_empirical_samples.csv', 'csv')
    _check(list(empirical.columns) == ['n', 'date', 'date_type', 'equal_weight'] + cols,
           'empirical sample TAZ ordering')
    _check(empirical.n.is_unique, 'duplicate empirical sample IDs')
    data = dict(I=I, date_types=EVENTS, N_total=N, N_type={}, N_valid={}, date_prob={},
                valid_samples={}, demand_hat={}, mu_hat=dict(zip(I, map(float, mu))),
                sigma_tilde={(i,j): float(sigma[a,b]) for a,i in enumerate(I) for b,j in enumerate(I)})
    global_to_solver, solver_to_global, event_ids, valid_ids = {}, {}, {}, {}
    expected_empirical_ids = []
    for l in EVENTS:
        all_ids = samples.loc[samples.s_n == l, 'n'].tolist()
        kept = membership.loc[(membership.date_type == l) & membership.inside_omega, 'n'].tolist()
        nl, nv = len(all_ids), len(kept)
        _check(nv > 0, l + ': empty support sample set')
        q = nl/N
        for table in tables:
            row = table.set_index('date_type').loc[l]
            _check(row.N_l == nl, l + ': N_l mismatch')
            _close(row.q_l, q, l + ': q_l must use pre-filter counts')
        _check(json.loads(indices.loc[l, 'global_ids_N_l']) == all_ids, l + ': event index set mismatch')
        row = event_support.loc[l]
        _check(json.loads(row.global_ids_N_l) == all_ids and json.loads(row.global_ids_I_l_omega) == kept,
               l + ': support IDs mismatch')
        _check(row.N_l_omega == nv and set(kept).issubset(all_ids), l + ': support count mismatch')
        _close(row.equal_weight, 1/nv, l + ': empirical weight')
        _close(row.support_retention_rate, nv/nl, l + ': retention rate')
        data['N_type'][l], data['N_valid'][l], data['date_prob'][l] = nl, nv, float(stats.loc[l,'q_l'])
        event_ids[l], valid_ids[l] = tuple(all_ids), tuple(kept)
        for local, global_id in enumerate(all_ids, 1):
            global_to_solver[global_id] = (l, local)
            solver_to_global[l, local] = global_id
        data['valid_samples'][l] = tuple(global_to_solver[n][1] for n in kept)
        part = empirical.loc[empirical.date_type == l]
        _check(list(part.n) == kept, l + ': empirical IDs mismatch')
        _check(list(part.date) == samples.iloc[np.array(kept)-1].date.tolist(), l + ': empirical dates')
        _close(part.equal_weight.to_numpy(), 1/nv, l + ': empirical weights')
        _close(part[cols].to_numpy(), D[np.array(kept)-1], l + ': empirical demand differs', atol=0)
        expected_empirical_ids.extend(kept)
        for n in kept:
            local = global_to_solver[n][1]
            for k, i in enumerate(I):
                data['demand_hat'][l, local, i] = float(D[n-1,k])
    _check(list(empirical.n) == expected_empirical_ids, 'empirical extra/missing/reordered samples')
    _check(sum(data['N_type'].values()) == N, 'sum N_l != N')
    _close(sum(data['date_prob'].values()), 1., 'sum q_l != 1')
    epsilon_artifact = read(root3 / 'epsilon_status.json')
    _check(epsilon_artifact['epsilon_status'] == m3['epsilon_status'], 'epsilon manifest mismatch')
    params = dict(support_radius=r, r_squared=r2)
    pending, incompatibilities = [], []
    radius_dir = root3.parent / 'wasserstein_radius_calibration_outputs'
    final_radius = radius_dir / 'final_radius_calibration.json'
    radius_metadata = radius_dir / 'run_metadata.json'
    calibrated_radius_artifact = None
    if final_radius.is_file() or radius_metadata.is_file():
        _check(final_radius.is_file() and radius_metadata.is_file(),
               'incomplete Wasserstein calibration artifact')
        calibrated_radius_artifact = read(final_radius)
        calibration_metadata = read(radius_metadata)
        _check(calibrated_radius_artifact.get('status') == calibration_metadata.get('status') == 'SUCCESS',
               'Wasserstein calibration artifact is not SUCCESS')
        _check(calibration_metadata.get('demand_dimension') == d,
               'Wasserstein calibration dimension mismatch')
        _check(calibration_metadata.get('input_module3_manifest_sha256') ==
               hashlib.sha256((root3 / 'run_manifest.json').read_bytes()).hexdigest(),
               'Wasserstein calibration references a different module3 manifest')
        _check(calibration_metadata.get('input_empirical_samples_sha256') ==
               hashlib.sha256((root3 / 'eventwise_empirical_samples.csv').read_bytes()).hexdigest(),
               'Wasserstein calibration references different empirical samples')
        _check(calibration_metadata.get('input_eventwise_support_sha256') ==
               hashlib.sha256((root3 / 'eventwise_support.csv').read_bytes()).hexdigest(),
               'Wasserstein calibration references different event support')
        _check(calibration_metadata.get('input_support_membership_sha256') ==
               hashlib.sha256((root3 / 'support_membership.csv').read_bytes()).hexdigest(),
               'Wasserstein calibration references different support membership')
        _check(calibrated_radius_artifact.get('N_l_Omega') == {l: data['N_valid'][l] for l in EVENTS},
               'Wasserstein calibration N_l_Omega mismatch')
        c = _number(calibrated_radius_artifact.get('c_epsilon'), 'calibrated c_epsilon', positive=True)
        hats = {l: _number(calibrated_radius_artifact.get('c_hat_' + l), 'c_hat_' + l, positive=True)
                for l in EVENTS}
        _close(c, max(hats.values()), 'calibrated c_epsilon must equal max c_hat_l')
        expected = {l: c * data['N_valid'][l]**(-1.0/d) for l in EVENTS}
        observed = {l: _number(calibrated_radius_artifact.get('epsilon_' + l), 'epsilon_' + l, positive=True)
                    for l in EVENTS}
        for l in EVENTS:
            _close(observed[l], expected[l], l + ': calibrated epsilon formula mismatch')
        _check(c_epsilon is None, 'explicit c_epsilon cannot override a formal calibration artifact')
        params['wasserstein_radius'] = observed
    elif c_epsilon is None:
        pending.append('c_epsilon (intentionally pending); wasserstein_radius omitted')
    else:
        c = _number(c_epsilon, 'c_epsilon')
        params['wasserstein_radius'] = {l: c * data['N_valid'][l]**(-1.0/d) for l in EVENTS}
        if c == 0:
            incompatibilities.append('solver requires epsilon_l > 0; explicit c_epsilon=0 gives zero')
    if r == 0:
        incompatibilities.append('solver requires support_radius > 0; artifact r=0')
    business = {} if business_input is None else dict(business_input)
    _check(not (set(business)-set(REQUIRED_BUSINESS)), 'unknown BUSINESS_INPUT keys; no derived/statistical override allowed')
    missing = [name for name in REQUIRED_BUSINESS if name not in business or business[name] is None]
    if 'J' not in missing:
        J = tuple(business['J'])
        _check(bool(J) and len(set(J)) == len(J) and all(type(j) in (int,str) for j in J), 'J invalid')
        data['J'] = J
    for name in SCALARS:
        if name not in missing:
            params[name] = _number(business[name], name, positive=name in (
                'rho','shortage_penalty','fixed_capacity_per_unit','mobile_capacity_per_unit'))
    for name in MAPPINGS:
        if name in missing:
            continue
        if name == 'fixed_count_ub':
            keys = I
        elif 'J' not in data:
            continue
        else:
            keys = data['J'] if name in ('hub_open_cost','mobile_fleet_ub') else tuple(product(I, data['J']))
        _check(set(business[name]) == set(keys), name + ': canonical index keys mismatch')
        mapped = {key: _number(business[name][key], name) for key in keys}
        (data if name == 'distance' else params)[name] = mapped
    if 'distance' in data and 'transport_cost_per_distance' in params:
        data['transport_cost'] = {key: 2*params['transport_cost_per_distance']*v for key,v in data['distance'].items()}
    options = {} if solver_options is None else dict(solver_options)
    _check(set(options).issubset({'separation_tolerance','gurobi','separation_gurobi',
                                  'optimized_separation_gurobi'}), 'unknown solver options')
    if 'separation_tolerance' in options:
        options['separation_tolerance'] = _number(options['separation_tolerance'], 'separation_tolerance')
    else:
        incompatibilities.append('CG requires explicit separation_tolerance (or tolerance at direct call); no production setting frozen')
    params.update(options)
    big_m_fields = {'mobile_fleet_ub', 'deployment_ub'}
    missing_big_m = [name for name in missing if name in big_m_fields]
    missing_other_business = [name for name in missing if name not in big_m_fields]
    blockers = {'missing_business_input': missing_other_business, 'missing_big_m': missing_big_m,
                'interface_inconsistencies': incompatibilities, 'pending': pending}
    status = ('BLOCKED_BY_INTERFACE_INCONSISTENCY' if incompatibilities else
              'BLOCKED_BY_MISSING_BUSINESS_INPUT' if missing_other_business else
              'BLOCKED_BY_BIG_M' if missing_big_m else
              'READY_EXCEPT_C_EPSILON' if pending else 'READY_FOR_REAL_DATA_SMOKE_TEST')
    return SolverInputs(data, params, dict(status=status, blockers=blockers,
        artifact_validation='PASS', global_to_solver=global_to_solver, solver_to_global=solver_to_global,
        event_global_ids=event_ids, support_global_ids=valid_ids, D_hat=D, mu=mu,
        Sigma_hat=sigma_hat, Sigma_tilde=sigma, chol_L=L, source_sha256=hashes,
        epsilon_artifact=epsilon_artifact, calibrated_radius_artifact=calibrated_radius_artifact,
        c_epsilon=(calibrated_radius_artifact or {}).get('c_epsilon', c_epsilon), solver_called=False))


def build_frozen_real_case_solver_inputs(module2_dir, module3_dir, *, c_epsilon=None,
                                         solver_options=None):
    """唯一的真实案例便捷入口：业务配置先加载，再进入同一 adapter。"""
    from real_case_business import build_real_case_business_inputs

    order_path = Path(module2_dir) / 'taz_vector_order.csv'
    order = tuple(int(value) for value in pd.read_csv(order_path)['taz_id'])
    prepared = build_real_case_business_inputs(order)
    options = prepared.solver_options if solver_options is None else solver_options
    result = build_solver_inputs(module2_dir, module3_dir, prepared.business_input,
                                 c_epsilon=c_epsilon, solver_options=options)
    result.metadata['business_audit'] = prepared.audit
    return result
