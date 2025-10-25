from flask import Flask, request, jsonify
from flask_cors import CORS
import quant_risk_engine
import traceback
from typing import Dict, Any, Optional, List

app = Flask(__name__)
CORS(app)

DEFAULT_VAR_SIMULATIONS = 10000
DEFAULT_VAR_CONFIDENCE = 0.95
DEFAULT_VAR_TIME_HORIZON = 1.0

def validate_portfolio_item(item: Dict[str, Any], index: int) -> None:
    required_fields = ['type', 'strike', 'expiry', 'asset_id', 'quantity']
    for field in required_fields:
        if field not in item:
            raise ValueError(f"Portfolio item {index}: missing required field '{field}'")
    
    option_style = item.get('style', 'european').lower()
    if option_style not in ['european', 'american']:
        raise ValueError(f"Portfolio item {index}: style must be 'european' or 'american'")
    
    if item['type'].lower() not in ['call', 'put']:
        raise ValueError(f"Portfolio item {index}: type must be 'call' or 'put'")
    
    if not isinstance(item['strike'], (int, float)) or item['strike'] <= 0:
        raise ValueError(f"Portfolio item {index}: strike must be a positive number")
    
    if not isinstance(item['expiry'], (int, float)) or item['expiry'] <= 0:
        raise ValueError(f"Portfolio item {index}: expiry must be a positive number")
    
    if not isinstance(item['quantity'], int):
        raise ValueError(f"Portfolio item {index}: quantity must be an integer")
    
    if not isinstance(item['asset_id'], str) or not item['asset_id'].strip():
        raise ValueError(f"Portfolio item {index}: asset_id must be a non-empty string")
    
    pricing_model = item.get('pricing_model', 'blackscholes').lower()
    if pricing_model not in ['blackscholes', 'binomial', 'jumpdiffusion']:
        raise ValueError(f"Portfolio item {index}: pricing_model must be 'blackscholes', 'binomial', or 'jumpdiffusion'")

def validate_market_data(asset_id: str, md: Dict[str, Any]) -> None:
    required_fields = ['spot', 'rate', 'vol']
    for field in required_fields:
        if field not in md:
            raise ValueError(f"Market data for '{asset_id}': missing required field '{field}'")
    
    if not isinstance(md['spot'], (int, float)) or md['spot'] <= 0:
        raise ValueError(f"Market data for '{asset_id}': spot must be a positive number")
    
    if not isinstance(md['rate'], (int, float)):
        raise ValueError(f"Market data for '{asset_id}': rate must be a number")
    
    if not isinstance(md['vol'], (int, float)) or md['vol'] < 0:
        raise ValueError(f"Market data for '{asset_id}': vol must be a non-negative number")
    
    if 'dividend' in md:
        if not isinstance(md['dividend'], (int, float)) or md['dividend'] < 0:
            raise ValueError(f"Market data for '{asset_id}': dividend must be a non-negative number")

def validate_var_parameters(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    validated = {
        'simulations': DEFAULT_VAR_SIMULATIONS,
        'confidence': DEFAULT_VAR_CONFIDENCE,
        'time_horizon': DEFAULT_VAR_TIME_HORIZON,
        'seed': None
    }
    
    if params is None:
        return validated
    
    if 'simulations' in params:
        sims = params['simulations']
        if not isinstance(sims, int) or sims <= 0 or sims > 1000000:
            raise ValueError("VaR simulations must be a positive integer <= 1,000,000")
        validated['simulations'] = sims
    
    if 'confidence' in params:
        conf = params['confidence']
        if not isinstance(conf, (int, float)) or conf <= 0.0 or conf >= 1.0:
            raise ValueError("VaR confidence must be between 0 and 1")
        validated['confidence'] = float(conf)
    
    if 'time_horizon' in params:
        horizon = params['time_horizon']
        if not isinstance(horizon, (int, float)) or horizon <= 0.0 or horizon > 252.0:
            raise ValueError("VaR time horizon must be between 0 and 252 days")
        validated['time_horizon'] = float(horizon)
    
    if 'seed' in params:
        seed = params['seed']
        if seed is not None:
            if not isinstance(seed, int) or seed < 0:
                raise ValueError("Random seed must be a non-negative integer")
            validated['seed'] = seed
    
    return validated

def create_option(item: Dict[str, Any]) -> Any:
    option_type = (quant_risk_engine.OptionType.Call 
                  if item['type'].lower() == 'call' 
                  else quant_risk_engine.OptionType.Put)
    
    option_style = item.get('style', 'european').lower()
    
    if option_style == 'american':
        binomial_steps = item.get('binomial_steps', 100)
        option = quant_risk_engine.AmericanOption(
            option_type,
            float(item['strike']),
            float(item['expiry']),
            item['asset_id'].strip(),
            binomial_steps
        )
    else:
        pricing_model_str = item.get('pricing_model', 'blackscholes').lower()
        
        if pricing_model_str == 'binomial':
            pricing_model = quant_risk_engine.PricingModel.Binomial
        elif pricing_model_str == 'jumpdiffusion':
            pricing_model = quant_risk_engine.PricingModel.MertonJumpDiffusion
        else:
            pricing_model = quant_risk_engine.PricingModel.BlackScholes
        
        option = quant_risk_engine.EuropeanOption(
            option_type,
            float(item['strike']),
            float(item['expiry']),
            item['asset_id'].strip(),
            pricing_model
        )
        
        if pricing_model_str == 'binomial':
            binomial_steps = item.get('binomial_steps', 100)
            option.set_binomial_steps(binomial_steps)
        
        if pricing_model_str == 'jumpdiffusion':
            jump_params = item.get('jump_parameters', {})
            lambda_val = jump_params.get('lambda', 2.0)
            jump_mean = jump_params.get('mean', -0.05)
            jump_vol = jump_params.get('vol', 0.15)
            option.set_jump_parameters(lambda_val, jump_mean, jump_vol)
    
    return option

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'quant-risk-engine',
        'version': '3.0',
        'features': [
            'european_options',
            'american_options',
            'multiple_pricing_models',
            'jump_diffusion',
            'portfolio_analytics',
            'var_calculation'
        ]
    }), 200

@app.route('/calculate_risk', methods=['POST'])
def calculate_risk():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request body must be valid JSON'}), 400
        
        if 'portfolio' not in data:
            return jsonify({'error': "Missing required field 'portfolio'"}), 400
        
        if 'market_data' not in data:
            return jsonify({'error': "Missing required field 'market_data'"}), 400

        portfolio_data = data['portfolio']
        market_data_map_py = data['market_data']
        var_params = data.get('var_parameters', None)
        
        if not isinstance(portfolio_data, list):
            return jsonify({'error': "Field 'portfolio' must be an array"}), 400
        
        if not isinstance(market_data_map_py, dict):
            return jsonify({'error': "Field 'market_data' must be an object"}), 400
        
        if len(portfolio_data) == 0:
            return jsonify({'error': 'Portfolio cannot be empty'}), 400
        
        if len(market_data_map_py) == 0:
            return jsonify({'error': 'Market data cannot be empty'}), 400

        for idx, item in enumerate(portfolio_data):
            validate_portfolio_item(item, idx)
        
        for asset_id, md in market_data_map_py.items():
            validate_market_data(asset_id, md)
        
        portfolio_assets = set(item['asset_id'] for item in portfolio_data)
        market_data_assets = set(market_data_map_py.keys())
        missing_assets = portfolio_assets - market_data_assets
        
        if missing_assets:
            return jsonify({
                'error': f"Missing market data for assets: {', '.join(missing_assets)}"
            }), 400

        var_config = validate_var_parameters(var_params)

        portfolio = quant_risk_engine.Portfolio()
        portfolio.reserve(len(portfolio_data))
        
        for item in portfolio_data:
            option = create_option(item)
            portfolio.add_instrument(option, item['quantity'])

        market_data_map_cpp = {}
        for asset_id, md_py in market_data_map_py.items():
            dividend = md_py.get('dividend', 0.0)
            md_cpp = quant_risk_engine.MarketData(
                asset_id,
                float(md_py['spot']),
                float(md_py['rate']),
                float(md_py['vol']),
                float(dividend)
            )
            market_data_map_cpp[asset_id] = md_cpp

        engine = quant_risk_engine.RiskEngine()
        engine.set_var_simulations(var_config['simulations'])
        engine.set_var_confidence_level(var_config['confidence'])
        engine.set_var_time_horizon_days(var_config['time_horizon'])
        
        if var_config['seed'] is not None:
            engine.set_random_seed(var_config['seed'])
            engine.set_use_fixed_seed(True)
        
        result_cpp = engine.calculate_portfolio_risk(portfolio, market_data_map_cpp)
        
        if not result_cpp.is_valid():
            return jsonify({'error': 'Risk calculation produced invalid results'}), 500

        result_py = {
            'total_pv': result_cpp.total_pv,
            'total_delta': result_cpp.total_delta,
            'total_gamma': result_cpp.total_gamma,
            'total_vega': result_cpp.total_vega,
            'total_theta': result_cpp.total_theta,
            'value_at_risk_95': result_cpp.value_at_risk_95,
            'portfolio_size': len(portfolio),
            'var_parameters': {
                'simulations': var_config['simulations'],
                'confidence_level': var_config['confidence'],
                'time_horizon_days': var_config['time_horizon']
            }
        }
        return jsonify(result_py), 200

    except ValueError as e:
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Runtime error: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/price_option', methods=['POST'])
def price_option():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request body must be valid JSON'}), 400
        
        required_fields = ['type', 'strike', 'expiry', 'asset_id', 'market_data']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f"Missing required field '{field}'"}), 400
        
        validate_market_data(data['asset_id'], data['market_data'])
        
        option = create_option({
            'type': data['type'],
            'strike': data['strike'],
            'expiry': data['expiry'],
            'asset_id': data['asset_id'],
            'style': data.get('style', 'european'),
            'pricing_model': data.get('pricing_model', 'blackscholes'),
            'binomial_steps': data.get('binomial_steps', 100),
            'jump_parameters': data.get('jump_parameters', {}),
            'quantity': 1
        })
        
        md_data = data['market_data']
        md = quant_risk_engine.MarketData(
            data['asset_id'],
            float(md_data['spot']),
            float(md_data['rate']),
            float(md_data['vol']),
            float(md_data.get('dividend', 0.0))
        )
        
        result = {
            'price': option.price(md),
            'delta': option.delta(md),
            'gamma': option.gamma(md),
            'vega': option.vega(md),
            'theta': option.theta(md),
            'instrument_type': option.get_instrument_type()
        }
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Runtime error: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/portfolio/net_position/<asset_id>', methods=['POST'])
def get_net_position(asset_id):
    try:
        data = request.get_json()
        
        if not data or 'portfolio' not in data:
            return jsonify({'error': 'Missing portfolio data'}), 400
        
        portfolio_data = data['portfolio']
        
        if not isinstance(portfolio_data, list):
            return jsonify({'error': 'Portfolio must be an array'}), 400
        
        for idx, item in enumerate(portfolio_data):
            validate_portfolio_item(item, idx)
        
        portfolio = quant_risk_engine.Portfolio()
        
        for item in portfolio_data:
            option = create_option(item)
            portfolio.add_instrument(option, item['quantity'])
        
        net_quantity = portfolio.get_total_quantity(asset_id)
        
        return jsonify({
            'asset_id': asset_id,
            'net_quantity': net_quantity,
            'direction': 'long' if net_quantity > 0 else ('short' if net_quantity < 0 else 'flat')
        }), 200
        
    except ValueError as e:
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Runtime error: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/portfolio/summary', methods=['POST'])
def portfolio_summary():
    try:
        data = request.get_json()

        if not data or 'portfolio' not in data:
            return jsonify({'error': 'Missing portfolio data'}), 400

        portfolio_data = data['portfolio']

        if not isinstance(portfolio_data, list):
            return jsonify({'error': 'Portfolio must be an array'}), 400

        for idx, item in enumerate(portfolio_data):
            validate_portfolio_item(item, idx)

        portfolio = quant_risk_engine.Portfolio()

        for item in portfolio_data:
            option = create_option(item)
            portfolio.add_instrument(option, item['quantity'])

        assets = set(item['asset_id'] for item in portfolio_data)
        net_positions = {}
        for asset in assets:
            net_positions[asset] = portfolio.get_total_quantity(asset)

        instrument_counts = {
            'european': sum(1 for item in portfolio_data if item.get('style', 'european') == 'european'),
            'american': sum(1 for item in portfolio_data if item.get('style', 'european') == 'american'),
            'calls': sum(1 for item in portfolio_data if item['type'].lower() == 'call'),
            'puts': sum(1 for item in portfolio_data if item['type'].lower() == 'put')
        }

        return jsonify({
            'portfolio_size': len(portfolio),
            'unique_assets': len(assets),
            'net_positions': net_positions,
            'instrument_counts': instrument_counts
        }), 200

    except ValueError as e:
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Runtime error: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/analytics/greeks_time_series', methods=['POST'])
def greeks_time_series():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Request body must be valid JSON'}), 400

        required_fields = ['portfolio', 'market_data', 'time_points']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        portfolio_data = data['portfolio']
        market_data_map_py = data['market_data']
        time_points = data['time_points']

        if not isinstance(portfolio_data, list) or len(portfolio_data) == 0:
            return jsonify({'error': 'Portfolio must be a non-empty array'}), 400

        if not isinstance(market_data_map_py, dict) or len(market_data_map_py) == 0:
            return jsonify({'error': 'Market data must be a non-empty object'}), 400

        if not isinstance(time_points, list) or len(time_points) == 0:
            return jsonify({'error': 'Time points must be a non-empty array'}), 400

        # Validate time points
        for days in time_points:
            if not isinstance(days, (int, float)) or days <= 0:
                return jsonify({'error': 'Time points must be positive numbers'}), 400

        # Validate portfolio and market data
        for idx, item in enumerate(portfolio_data):
            validate_portfolio_item(item, idx)

        for asset_id, md in market_data_map_py.items():
            validate_market_data(asset_id, md)

        # Check asset coverage
        portfolio_assets = set(item['asset_id'] for item in portfolio_data)
        market_data_assets = set(market_data_map_py.keys())
        missing_assets = portfolio_assets - market_data_assets
        if missing_assets:
            return jsonify({'error': f'Missing market data for assets: {list(missing_assets)}'}), 400

        # Create portfolio
        portfolio = quant_risk_engine.Portfolio()
        portfolio.reserve(len(portfolio_data))

        for item in portfolio_data:
            option = create_option(item)
            portfolio.add_instrument(option, item['quantity'])

        # Convert market data to C++ format
        market_data_map_cpp = {}
        for asset_id, md_py in market_data_map_py.items():
            dividend = md_py.get('dividend', 0.0)
            md_cpp = quant_risk_engine.MarketData(
                asset_id,
                float(md_py['spot']),
                float(md_py['rate']),
                float(md_py['vol']),
                float(dividend)
            )
            market_data_map_cpp[asset_id] = md_cpp

        # Calculate time series
        time_series = []
        for days in time_points:
            # Create modified market data with adjusted time horizon for VaR
            # Note: Greeks are calculated at current time, VaR uses the specified horizon
            engine = quant_risk_engine.RiskEngine()
            engine.set_var_simulations(10000)  # Reduced for speed
            engine.set_var_confidence_level(0.95)
            engine.set_var_time_horizon_days(float(days))

            try:
                result = engine.calculate_portfolio_risk(portfolio, market_data_map_cpp)
                time_series.append({
                    'days': days,
                    'delta': result.total_delta,
                    'gamma': result.total_gamma,
                    'vega': result.total_vega,
                    'theta': result.total_theta,
                    'var': result.value_at_risk_95
                })
            except Exception as e:
                return jsonify({'error': f'Calculation failed for {days} days: {str(e)}'}), 500

        return jsonify({'time_series': time_series}), 200

    except ValueError as e:
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Runtime error: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/analytics/iv_surface', methods=['POST'])
def iv_surface():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Request body must be valid JSON'}), 400

        required_fields = ['portfolio', 'market_data']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        portfolio_data = data['portfolio']
        market_data_map_py = data['market_data']

        # Default ranges if not provided
        strike_range = data.get('strike_range', {'min': 50, 'max': 150, 'steps': 21})
        expiry_range = data.get('expiry_range', {'min': 0.1, 'max': 2.0, 'steps': 20})

        if not isinstance(portfolio_data, list) or len(portfolio_data) == 0:
            return jsonify({'error': 'Portfolio must be a non-empty array'}), 400

        if not isinstance(market_data_map_py, dict) or len(market_data_map_py) == 0:
            return jsonify({'error': 'Market data must be a non-empty object'}), 400

        # Validate ranges
        for range_obj in [strike_range, expiry_range]:
            if not all(key in range_obj for key in ['min', 'max', 'steps']):
                return jsonify({'error': 'Range objects must have min, max, and steps'}), 400
            if range_obj['min'] >= range_obj['max'] or range_obj['steps'] < 2:
                return jsonify({'error': 'Invalid range parameters'}), 400

        # For IV surface, we need at least one option in the portfolio
        if not any(item.get('pricing_model', 'blackscholes').lower() == 'blackscholes' for item in portfolio_data):
            return jsonify({'error': 'IV surface requires at least one Black-Scholes priced option'}), 400

        # Validate portfolio and market data
        for idx, item in enumerate(portfolio_data):
            validate_portfolio_item(item, idx)

        for asset_id, md in market_data_map_py.items():
            validate_market_data(asset_id, md)

        # Generate strike and expiry grids
        strikes = []
        step_size = (strike_range['max'] - strike_range['min']) / (strike_range['steps'] - 1)
        for i in range(strike_range['steps']):
            strikes.append(strike_range['min'] + i * step_size)

        expiries = []
        expiry_step = (expiry_range['max'] - expiry_range['min']) / (expiry_range['steps'] - 1)
        for i in range(expiry_range['steps']):
            expiries.append(expiry_range['min'] + i * expiry_step)

        # Calculate IV surface - for simplicity, use the first option
        first_option_data = next(item for item in portfolio_data if item.get('pricing_model', 'blackscholes').lower() == 'blackscholes')
        asset_id = first_option_data['asset_id']

        if asset_id not in market_data_map_py:
            return jsonify({'error': f'Market data missing for {asset_id}'}), 400

        md_py = market_data_map_py[asset_id]
        spot = md_py['spot']
        rate = md_py['rate']

        # Calculate IV for each strike/expiry combination
        iv_matrix = []
        for expiry in expiries:
            iv_row = []
            for strike in strikes:
                try:
                    # Create a temporary European option for IV calculation
                    option_type = quant_risk_engine.OptionType.Call if first_option_data['type'].lower() == 'call' else quant_risk_engine.OptionType.Put
                    temp_option = quant_risk_engine.EuropeanOption(option_type, strike, expiry, asset_id)

                    # Get market price (simplified - using Black-Scholes with some base vol)
                    market_price = temp_option.price(quant_risk_engine.MarketData(asset_id, spot, rate, 0.25))  # 25% base vol

                    # Calculate implied volatility
                    # Note: This is a simplified implementation
                    # Real IV calculation would use numerical methods
                    implied_vol = 0.25  # Placeholder - would implement proper IV calculation

                    iv_row.append(implied_vol)
                except Exception:
                    iv_row.append(None)  # Failed calculation
            iv_matrix.append(iv_row)

        return jsonify({
            'surface': {
                'strikes': strikes,
                'expiries': expiries,
                'iv_matrix': iv_matrix
            }
        }), 200

    except ValueError as e:
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except RuntimeError as e:
        return jsonify({'error': f'Runtime error: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')