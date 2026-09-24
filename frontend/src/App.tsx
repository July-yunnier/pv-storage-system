import { Layout, Menu, Typography } from 'antd'
import {
  DashboardOutlined,
  LineChartOutlined,
  ScheduleOutlined,
  FundOutlined,
  DatabaseOutlined,
} from '@ant-design/icons'
import {
  BrowserRouter,
  Routes,
  Route,
  useNavigate,
  useLocation,
} from 'react-router-dom'
import OverviewPage from './pages/OverviewPage'
import CurvesPage from './pages/CurvesPage'
import DispatchPage from './pages/DispatchPage'
import DispatchDetailPage from './pages/DispatchDetailPage'
import SizingPage from './pages/SizingPage'
import SizingDetailPage from './pages/SizingDetailPage'
import DataPage from './pages/DataPage'

const { Sider, Content } = Layout

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '站点总览' },
  { key: '/curves', icon: <LineChartOutlined />, label: '曲线分析' },
  { key: '/dispatch', icon: <ScheduleOutlined />, label: '调度计划' },
  { key: '/sizing', icon: <FundOutlined />, label: '方案分析' },
  { key: '/data', icon: <DatabaseOutlined />, label: '数据管理' },
]

function AppLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const selectedKey =
    menuItems.find(
      (item) =>
        item.key !== '/' && location.pathname.startsWith(item.key),
    )?.key ?? '/'

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        theme="dark"
        width={220}
        style={{
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          height: '100vh',
          zIndex: 100,
          overflow: 'auto',
        }}
      >
        <div
          style={{
            position: 'sticky',
            top: 0,
            zIndex: 1,
            padding: '20px 16px',
            background: '#001529',
          }}
        >
          <Typography.Title level={4} style={{ color: '#fff', margin: 0 }}>
            光储一体
          </Typography.Title>
          <Typography.Text style={{ color: 'rgba(255,255,255,0.65)', fontSize: 12 }}>
            储能配置优化系统
          </Typography.Text>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout style={{ marginLeft: 220 }}>
        <Content style={{ padding: 24, background: '#f5f5f5' }}>
          <div style={{ maxWidth: 1280, margin: '0 auto' }}>
            <Routes>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/curves" element={<CurvesPage />} />
              <Route path="/dispatch" element={<DispatchPage />} />
              <Route path="/dispatch/:runId" element={<DispatchDetailPage />} />
              <Route path="/sizing" element={<SizingPage />} />
              <Route path="/sizing/:runId" element={<SizingDetailPage />} />
              <Route path="/data" element={<DataPage />} />
            </Routes>
          </div>
        </Content>
      </Layout>
    </Layout>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  )
}
