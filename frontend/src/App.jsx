import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ReferenceLine
} from 'recharts';
import {
  Activity, AlertTriangle, CheckCircle, Info, Zap, Wind, Thermometer,
  Settings, ChevronDown, BarChart2, Cpu, Sliders, RefreshCw, Plane
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_Base = "http://localhost:8000";

// --- Components ---

const GlassCard = ({ children, className = "", delay = 0 }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay }}
    className={`bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-xl overflow-hidden ${className}`}
  >
    {children}
  </motion.div>
);

const KPICard = ({ label, value, sub, icon: Icon, color, delay }) => (
  <GlassCard delay={delay} className="p-6 relative overflow-hidden group">
    <div className={`absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity duration-500 text-${color}-500`}>
      <Icon size={120} />
    </div>
    <div className="relative z-10">
      <div className="flex items-center gap-3 mb-2 text-slate-400">
        <div className={`p-2 rounded-lg bg-${color}-500/10 text-${color}-400`}>
          <Icon size={20} />
        </div>
        <span className="text-sm font-medium tracking-wide uppercase">{label}</span>
      </div>
      <div className="text-3xl font-bold text-white mb-1 tracking-tight">
        {value}
      </div>
      <div className={`text-sm font-medium text-${color}-400`}>
        {sub}
      </div>
    </div>
  </GlassCard>
);

const SimulationPanel = ({ engineId, onClose }) => {
  const [adjustments, setAdjustments] = useState({ sensor_11: 1.0, sensor_4: 1.0, sensor_9: 1.0 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const runSimulation = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_Base}/predict_simulated`, {
        engine_id: engineId,
        adjustments: adjustments
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ x: 400 }} animate={{ x: 0 }} exit={{ x: 400 }}
      className="fixed right-0 top-0 bottom-0 w-96 bg-slate-900 border-l border-slate-700 p-6 z-50 shadow-2xl flex flex-col"
    >
      <div className="flex items-center justify-between mb-8">
        <h3 className="text-xl font-bold text-white flex items-center gap-2">
          <Sliders className="text-cyan-400" /> What-If Analysis
        </h3>
        <button onClick={onClose} className="text-slate-400 hover:text-white">✕</button>
      </div>

      <div className="space-y-6 flex-1 overflow-y-auto">
        <div className="space-y-4">
          <label className="text-sm font-medium text-slate-300">Fuel/Air Ratio Adjustment</label>
          <input
            type="range" min="0.9" max="1.1" step="0.01"
            value={adjustments.sensor_11}
            onChange={(e) => setAdjustments({ ...adjustments, sensor_11: parseFloat(e.target.value) })}
            className="w-full accent-cyan-500"
          />
          <div className="text-right text-xs text-cyan-400">{(adjustments.sensor_11 * 100).toFixed(0)}%</div>
        </div>

        <div className="space-y-4">
          <label className="text-sm font-medium text-slate-300">LPT Temp Adjustment</label>
          <input
            type="range" min="0.9" max="1.1" step="0.01"
            value={adjustments.sensor_4}
            onChange={(e) => setAdjustments({ ...adjustments, sensor_4: parseFloat(e.target.value) })}
            className="w-full accent-amber-500"
          />
          <div className="text-right text-xs text-amber-400">{(adjustments.sensor_4 * 100).toFixed(0)}%</div>
        </div>

        <button
          onClick={runSimulation}
          className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-bold transition-colors flex items-center justify-center gap-2"
        >
          {loading ? <RefreshCw className="animate-spin" /> : "Run Simulation"}
        </button>

        {result && (
          <div className="mt-6 bg-slate-800 p-4 rounded-xl border border-slate-700 animate-fade-in">
            <h4 className="text-sm font-semibold text-slate-400 mb-2">Projected Impact</h4>
            <div className="flex justify-between items-end mb-2">
              <span className="text-2xl font-bold text-white">{result.simulated_rul.toFixed(1)} <span className="text-sm text-slate-500">Cycles</span></span>
              <span className={`text-sm font-bold ${result.delta < 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                {result.delta > 0 ? '+' : ''}{result.delta.toFixed(1)}
              </span>
            </div>
            <div className="text-xs text-slate-500">
              New State: <span className="text-slate-300">{result.simulated_state}</span>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

const FleetSidebar = ({ engines, selected, onSelect }) => (
  <div className="hidden lg:flex w-64 flex-col bg-slate-900 border-r border-slate-800 h-screen sticky top-0 overflow-y-auto">
    <div className="p-6 border-b border-slate-800">
      <h2 className="text-lg font-bold text-white flex items-center gap-2">
        <Plane className="text-cyan-500" /> AeroFleet
      </h2>
      <p className="text-xs text-slate-500 mt-1">Global Asset Monitor</p>
    </div>
    <div className="p-4 space-y-2">
      {engines.map((id) => (
        <button
          key={id}
          onClick={() => onSelect(id)}
          className={`w-full text-left px-4 py-3 rounded-xl transition-all ${selected === id
            ? 'bg-cyan-500/10 border border-cyan-500/50 text-white'
            : 'text-slate-400 hover:bg-slate-800 hover:text-white'
            }`}
        >
          <div className="flex justify-between items-center">
            <span className="font-mono font-medium">Unit #{id}</span>
            <div className={`w-2 h-2 rounded-full ${id % 3 === 0 ? 'bg-amber-500' : 'bg-emerald-500'}`}></div>
          </div>
        </button>
      ))}
    </div>
  </div>
);

// --- Main App ---

function App() {
  const [engines, setEngines] = useState([]);
  const [selectedEngine, setSelectedEngine] = useState(null);
  const [engineData, setEngineData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [showSim, setShowSim] = useState(false);

  useEffect(() => {
    fetchEngines();
  }, []);

  useEffect(() => {
    if (selectedEngine) {
      fetchEngineData(selectedEngine);
      fetchPrediction(selectedEngine);
    }
  }, [selectedEngine]);

  const fetchEngines = async () => {
    try {
      const res = await axios.get(`${API_Base}/engines`);
      setEngines(res.data.engine_ids);
      if (res.data.engine_ids.length > 0) setSelectedEngine(res.data.engine_ids[0]);
    } catch (err) {
      console.error("Failed to fetch engines", err);
    }
  };

  const fetchEngineData = async (id) => {
    try {
      const res = await axios.get(`${API_Base}/engine/${id}`);
      setEngineData(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchPrediction = async (id) => {
    try {
      const res = await axios.post(`${API_Base}/predict_explain`, { engine_id: id });
      setPrediction(res.data.prediction);
      setExplanation(res.data.explanation);
    } catch (err) {
      console.error(err);
    }
  };

  if (!selectedEngine) return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center text-slate-500">
      <RefreshCw className="animate-spin" />
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans flex">
      <FleetSidebar engines={engines} selected={selectedEngine} onSelect={setSelectedEngine} />

      <div className="flex-1 relative">
        {/* Top Navbar */}
        <nav className="sticky top-0 z-40 bg-slate-950/80 backdrop-blur-md border-b border-slate-800/60 px-8 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-xl font-bold text-white">Engine Health Dashboard</h1>
            <p className="text-xs text-slate-500">Unit #{selectedEngine} • Gas Turbine Cycle Monitor</p>
          </div>

          <button
            onClick={() => setShowSim(!showSim)}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-white font-medium transition-colors border border-slate-700"
          >
            <Sliders size={16} /> Simulation Mode
          </button>
        </nav>

        <main className="p-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <KPICard
              label="Est. Flight Cycles (EFL)"
              value={prediction ? `${prediction.rul.toFixed(1)}` : "..."}
              sub="Prognostic Estimate"
              icon={Wind}
              color="cyan"
              delay={0.1}
            />
            <KPICard
              label="Health Status"
              value={prediction ? prediction.state : "..."}
              sub="Flight Worthiness"
              icon={Activity}
              color={prediction?.state === 'Normal' ? 'emerald' : 'rose'}
              delay={0.2}
            />
            <KPICard
              label="Total Cycles"
              value={engineData ? engineData.current_cycle : "..."}
              sub="Since New"
              icon={RefreshCw}
              color="slate"
              delay={0.3}
            />
            <KPICard
              label="Confidence Score"
              value={prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : "..."}
              sub="Model Accuracy"
              icon={BarChart2}
              color="violet"
              delay={0.4}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Insight Panel */}
            <GlassCard className="col-span-1 lg:col-span-1 p-6" delay={0.5}>
              <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
                <Cpu className="text-cyan-400" size={20} /> Failure Forensics
              </h3>
              {explanation ? (
                <>
                  <div className="h-64 mb-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart layout="vertical" data={explanation.lime_features.slice(0, 6)} margin={{ left: 10 }}>
                        <XAxis type="number" hide />
                        <YAxis type="category" dataKey="feature" width={110} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                        <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#fff' }} />
                        <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
                          {explanation.lime_features.slice(0, 6).map((e, i) => (
                            <Cell key={i} fill={e.weight > 0 ? '#ef4444' : '#10b981'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                    <h4 className="text-xs font-bold text-cyan-400 uppercase mb-2">Technician Advisory</h4>
                    <p className="text-sm text-slate-300 leading-relaxed">
                      {explanation.summary}
                    </p>
                  </div>
                </>
              ) : <div className="text-slate-500">Loading Forensics...</div>}
            </GlassCard>

            {/* Charts */}
            <GlassCard className="col-span-1 lg:col-span-2 p-6" delay={0.6}>
              <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
                <Activity className="text-amber-400" size={20} /> Telemetry Trends
              </h3>
              {engineData ? (
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={engineData.history.slice(-100)}>
                      <defs>
                        <linearGradient id="colorP" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                      <XAxis dataKey="time_in_cycles" stroke="#64748b" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
                      <YAxis stroke="#64748b" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                      <Area type="monotone" dataKey="sensor_11" name="Static Pressure" stroke="#ef4444" strokeWidth={2} fill="url(#colorP)" />
                      <Area type="monotone" dataKey="sensor_4" name="LPT Temp" stroke="#f59e0b" strokeWidth={2} fill="transparent" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : null}
            </GlassCard>
          </div>
        </main>

        <AnimatePresence>
          {showSim && <SimulationPanel engineId={selectedEngine} onClose={() => setShowSim(false)} />}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;
