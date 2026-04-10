import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area 
} from 'recharts';
import { 
  LayoutDashboard, ShoppingBag, Search, Bell, AlertTriangle, Users, Star, Box, Check
} from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:8002/api';

function Sidebar({ activeTab, setActiveTab }) {
  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        Williams Sonoma AI
      </div>
      <nav>
        <div 
          className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          <LayoutDashboard className="nav-icon" size={20} />
          Merchandise Engine
        </div>
        <div 
          className={`nav-link ${activeTab === 'customer' ? 'active' : ''}`}
          onClick={() => setActiveTab('customer')}
        >
          <ShoppingBag className="nav-icon" size={20} />
          Smart Shopping
        </div>
      </nav>
    </div>
  );
}

function TopHeader() {
  return (
    <div className="header">
      <div style={{ position: 'relative', width: '300px' }}>
        <Search className="nav-icon" size={18} style={{ position: 'absolute', left: 12, top: 10, color: '#aaa' }} />
        <input 
          type="text" 
          placeholder="Search products..." 
          style={{ width: '100%', padding: '10px 10px 10px 38px', borderRadius: '20px', border: '1px solid #ddd' }}
        />
      </div>
      <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
        <Bell size={20} color="#777" />
        <div className="profile-circle">WS</div>
      </div>
    </div>
  );
}

function MerchandiseDashboard({ stats, categories, actions, scores }) {
  const [productSpec, setProductSpec] = useState({ price: 199.99, category: "Sofas", material: "Velvet" });
  const [predScore, setPredScore] = useState(null);

  const predictProduct = async () => {
    try {
      const { data } = await axios.post(`${API_BASE}/ml/predict_product`, productSpec);
      setPredScore(data);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', color: 'var(--secondary)' }}>Return Intelligence & ML</h1>
        <p style={{ color: 'var(--text-light)' }}>Live extraction from Returns.csv and Reviews.csv</p>
      </div>

      <div className="grid-4" style={{ marginBottom: '2rem' }}>
        <div className="card">
          <div className="card-header">Overall Return Rate</div>
          <div className="stat-value">{stats.return_rate}%</div>
          <div className="trend-down">Processed vs total orders</div>
        </div>
        <div className="card">
          <div className="card-header">Margin Erosion Alert</div>
          <div className="stat-value" style={{ color: 'var(--accent)' }}>${stats.margin_erosion.toLocaleString()}</div>
          <div className="trend-down">Total return costs mapped</div>
        </div>
        <div className="card" style={{ gridColumn: 'span 2' }}>
          <div className="card-header">Predict New Product Success</div>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <input type="number" value={productSpec.price} onChange={e => setProductSpec({...productSpec, price: Number(e.target.value)})} style={{ width: '100px', padding: '8px', border: '1px solid #ddd', borderRadius: '8px' }} placeholder="Price" />
            <input type="text" value={productSpec.category} onChange={e => setProductSpec({...productSpec, category: e.target.value})} style={{ flex: 1, padding: '8px', border: '1px solid #ddd', borderRadius: '8px' }} placeholder="Category" />
            <input type="text" value={productSpec.material} onChange={e => setProductSpec({...productSpec, material: e.target.value})} style={{ flex: 1, padding: '8px', border: '1px solid #ddd', borderRadius: '8px' }} placeholder="Material" />
            <button className="btn btn-primary" onClick={predictProduct}>Predict Rating</button>
          </div>
          {predScore && (
            <div style={{ marginTop: '10px', fontSize: '0.9rem', color: '#27ae60', fontWeight: 'bold' }}>
              Projected Score: {predScore.predicted_score} / 5.0  (Confidence: {predScore.confidence}%)
            </div>
          )}
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: '2rem' }}>
        <div className="card">
          <div className="card-header">Priority Action List (NLP Generated)</div>
          <p style={{ fontSize: '0.85rem', color: '#777', marginBottom: '1rem' }}>Sourced dynamically from grouping return text reasons & negative reviews.</p>
          {actions.map((act, idx) => (
            <div className="list-item" key={idx}>
              <AlertTriangle color={act.severity === 'High' ? 'var(--accent)' : '#f39c12'} size={20} style={{ marginRight: 12 }} />
              <div className="list-item-content">
                <div className="list-item-title">{act.title}</div>
                <div className="list-item-subtitle">{act.subtitle}</div>
              </div>
              <div className="btn" style={{ padding: '6px 12px', fontSize: '13px', background: 'var(--primary)', color: 'white' }}>{act.button}</div>
            </div>
          ))}
        </div>

        <div className="card">
          <div className="card-header">Avg Return Score vs Time</div>
          <div style={{ height: 260 }}>
            {scores.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={scores}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="month" fontSize={12} />
                  <YAxis domain={[60, 100]} fontSize={12} />
                  <Tooltip />
                  <Area type="monotone" dataKey="score" stroke="var(--primary-dark)" fill="var(--primary)" fillOpacity={0.2} />
                </AreaChart>
              </ResponsiveContainer>
            ) : <p>Loading chart data...</p>}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">Top Returned Categories</div>
        <div style={{ height: 300 }}>
          {categories.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categories} layout="vertical" margin={{ left: 50 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" fontSize={13} />
                <Tooltip />
                <Bar dataKey="returns" fill="var(--secondary)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <p>Loading categories...</p>}
        </div>
      </div>
    </div>
  );
}

function CustomerOnboarding({ setPreferences }) {
  const [selected, setSelected] = useState([]);

  const toggle = (tag) => {
    if(selected.includes(tag)) setSelected(selected.filter(t => t !== tag));
    else setSelected([...selected, tag]);
  };

  const handleComplete = () => {
    // Save preferences
    setPreferences(selected);
  };

  return (
    <div className="onboarding-overlay">
      <div className="onboarding-modal">
        <h2 className="onboarding-title">Tell us about your space</h2>
        <p style={{ textAlign: 'center', color: '#666', marginBottom: '20px' }}>
          We use AI to personalize reviews and show you insights from buyers like you.
        </p>
        
        <div className="preference-options">
          {['Pet Owner', 'Family with Kids', 'Apartment', 'Large House', 'Bright Lighting', 'Warm Lighting', 'Minimalist', 'Cozy'].map(tag => (
            <button 
              key={tag} 
              className={`pref-btn ${selected.includes(tag) ? 'selected' : ''}`}
              onClick={() => toggle(tag)}
            >
              {tag} {selected.includes(tag) && <Check size={14} style={{display: 'inline', marginLeft: '4px'}}/>}
            </button>
          ))}
        </div>
        
        <button 
          className="btn btn-primary" 
          style={{ width: '100%', marginTop: '20px' }}
          onClick={handleComplete}
          disabled={selected.length === 0}
        >
          See My Personalized Experience
        </button>
      </div>
    </div>
  );
}

function CustomerView() {
  const [preferences, setPreferences] = useState(null);
  const [productInfo, setProductInfo] = useState(null);
  
  // Choose an existing product SKU to demonstrate
  const SKU_TO_FETCH = 'SKU00001'; 

  useEffect(() => {
    if (preferences) {
      axios.post(`${API_BASE}/customer/product`, {
        sku_id: SKU_TO_FETCH,
        tags: preferences
      })
      .then(res => setProductInfo(res.data))
      .catch(console.error);
    }
  }, [preferences]);

  return (
    <div>
      {!preferences && <CustomerOnboarding setPreferences={setPreferences} />}
      
      {productInfo && preferences ? (
        <>
          <div style={{ marginBottom: '2rem' }}>
            <h1 style={{ fontSize: '2rem', color: 'var(--secondary)', marginBottom: '8px' }}>{productInfo.product_name}</h1>
            <p style={{ color: 'var(--text-light)' }}>
              Tailored for you as a <b>{preferences.join(" + ")}</b> shopper. 
            </p>
          </div>

          <div className="product-hero">
            <div style={{ background: 'var(--card-bg)', border: '1px solid #eaeaea', flex: '0 0 350px', height: '350px', borderRadius: '12px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <Box size={64} color="#d1bfae" style={{ marginBottom: '16px' }} />
              <span style={{ color: '#aaa', fontSize: '2rem', fontWeight: 600 }}>${productInfo.price.toFixed(2)}</span>
              <span style={{ color: '#aaa', fontSize: '1rem', marginTop: '12px' }}>{productInfo.category}</span>
            </div>
            
            <div className="product-details">
              <div className="card" style={{ marginBottom: '1.5rem', background: 'faf9f7' }}>
                <div className="card-header">
                  <Star fill="#f1c40f" stroke="#f1c40f" size={18} style={{marginRight: '8px'}}/> 
                  Personalized AI Review Summary
                </div>
                <p style={{ fontSize: '1.1rem', marginBottom: '1.5rem', fontStyle: 'italic', lineHeight: '1.6' }}>
                  "{productInfo.ai_summary}"
                </p>
                <div className="tags-container" style={{ borderTop: '1px solid #eee', paddingTop: '1rem' }}>
                  <span style={{ fontSize: '0.85rem', color: '#666', marginRight: '8px', fontWeight: 'bold' }}>Key Tags:</span>
                  {productInfo.tags.map((tag, idx) => (
                    <span key={idx} className={`tag ${tag.type}`}>{tag.label}</span>
                  ))}
                </div>
              </div>

              <div className="card">
                <div className="card-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Users size={18} /> What Similar Users Think
                  </div>
                </div>
                
                {productInfo.similar_reviews.map((rev, idx) => (
                  <div className="list-item" key={idx} style={{ borderBottom: idx === productInfo.similar_reviews.length - 1 ? 'none' : '1px solid #eee', paddingBottom: '1rem', marginBottom: '1rem' }}>
                    <div>
                      <div style={{ display: 'flex', gap: '4px', color: '#f1c40f', marginBottom: '8px' }}>
                        {[...Array(rev.rating)].map((_, i) => <Star key={i} size={14} fill="currentColor" />)}
                      </div>
                      <b style={{ fontSize: '1.05rem', color: '#333' }}>{rev.title}</b>
                      <p style={{ fontSize: '0.95rem', color: '#555', marginTop: '6px', lineHeight: '1.5' }}>"{rev.text}"</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      ) : preferences ? (
        <div style={{display: 'flex', alignItems: 'center', gap:'12px', fontSize: '1.2rem'}}>
          <span className="spinner">⌛</span> AI is scanning 10,000+ reviews for you...
        </div>
      ) : null}
      
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [stats, setStats] = useState({ return_rate: 0, margin_erosion: 0 });
  const [categories, setCategories] = useState([]);
  const [actions, setActions] = useState([]);
  const [scores, setScores] = useState([]);

  useEffect(() => {
    // Initial fetches
    axios.get(`${API_BASE}/dashboard/stats`).then(res => setStats(res.data)).catch(console.error);
    axios.get(`${API_BASE}/dashboard/categories`).then(res => setCategories(res.data)).catch(console.error);
    axios.get(`${API_BASE}/dashboard/actions`).then(res => setActions(res.data)).catch(console.error);
    axios.get(`${API_BASE}/dashboard/scores`).then(res => setScores(res.data)).catch(console.error);
  }, []);

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="main-content">
        <TopHeader />
        {activeTab === 'dashboard' && <MerchandiseDashboard stats={stats} categories={categories} actions={actions} scores={scores}/>}
        {activeTab === 'customer' && <CustomerView />}
      </div>
    </div>
  );
}
